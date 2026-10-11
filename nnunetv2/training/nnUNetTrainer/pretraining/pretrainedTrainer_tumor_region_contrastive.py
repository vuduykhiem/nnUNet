"""
MAE finetuning trainer with tumor-vs-background region contrastive loss.

This trainer is intentionally isolated from the existing MAE finetuning trainer.
It subclasses PretrainedTrainer_250ep and changes only train_step behavior:

    total_loss = segmentation_loss + lambda * contrastive_loss

All annotated tumor voxels are treated as one abnormal/tumor class
(target > 0). This is not lytic-only and does not use real-vs-synthetic source
identity.

Default settings are conservative for a first ablation:

    contrastive_stage_index = 4
    contrastive_weight = 0.01
    contrastive_temperature = 0.1

Torch compile is disabled by default for this trainer because the auxiliary
loss depends on an intermediate encoder activation captured by a forward hook.
To experiment with compile after a successful smoke run, set:

    TUMOR_CONTRASTIVE_ALLOW_COMPILE=1
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer_variants import PretrainedTrainer_250ep
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


class _EncoderStageActivationCapture:
    """Capture one encoder stage output from a ResidualEncoderUNet/ResEncL."""

    def __init__(self, network: nn.Module, stage_index: int) -> None:
        self.stage_index = stage_index
        self.activation: Optional[torch.Tensor] = None
        self.handle = self._get_encoder_stage(network, stage_index).register_forward_hook(self._hook)

    @staticmethod
    def _unwrap_network(network: nn.Module) -> nn.Module:
        if hasattr(network, "module"):
            network = network.module
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
        return network

    def _get_encoder_stage(self, network: nn.Module, stage_index: int) -> nn.Module:
        network = self._unwrap_network(network)
        if not hasattr(network, "encoder") or not hasattr(network.encoder, "stages"):
            raise AttributeError("Expected network.encoder.stages for MAE-65 ResEncL.")
        if stage_index < 0 or stage_index >= len(network.encoder.stages):
            raise IndexError(f"encoder stage {stage_index} is outside available range 0..{len(network.encoder.stages) - 1}")
        return network.encoder.stages[stage_index]

    def _hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activation = output

    def close(self) -> None:
        self.handle.remove()


def _first_target_tensor(target: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    return target[0] if isinstance(target, list) else target


def _downsample_target_to_feature_grid(
    target: torch.Tensor | list[torch.Tensor],
    feature_spatial_shape: tuple[int, int, int],
    ignore_label: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_tensor = _first_target_tensor(target)
    if target_tensor.ndim != 5 or target_tensor.shape[1] != 1:
        raise ValueError(f"Expected target shape [B, 1, D, H, W], got {tuple(target_tensor.shape)}")

    target_lowres = F.interpolate(target_tensor.float(), size=feature_spatial_shape, mode="nearest")[:, 0]
    tumor_mask = target_lowres > 0
    background_mask = target_lowres == 0
    if ignore_label is not None:
        valid_mask = target_lowres != float(ignore_label)
        tumor_mask = tumor_mask & valid_mask
        background_mask = background_mask & valid_mask
    return tumor_mask, background_mask


def _sample_region_embeddings(
    feature_map: torch.Tensor,
    target: torch.Tensor | list[torch.Tensor],
    max_samples_per_class_per_case: int,
    min_samples_per_class_total: int,
    ignore_label: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if feature_map.ndim != 5:
        raise ValueError(f"Expected feature_map shape [B, C, d, h, w], got {tuple(feature_map.shape)}")

    batch_size, channels = feature_map.shape[:2]
    tumor_mask, background_mask = _downsample_target_to_feature_grid(
        target,
        feature_spatial_shape=tuple(feature_map.shape[2:]),
        ignore_label=ignore_label,
    )
    features_by_voxel = feature_map.permute(0, 2, 3, 4, 1).contiguous()

    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for b in range(batch_size):
        tumor_vectors = features_by_voxel[b][tumor_mask[b]]
        background_vectors = features_by_voxel[b][background_mask[b]]
        if tumor_vectors.numel() == 0 or background_vectors.numel() == 0:
            continue

        n_tumor = min(tumor_vectors.shape[0], max_samples_per_class_per_case)
        n_background = min(background_vectors.shape[0], max_samples_per_class_per_case)
        tumor_index = torch.randperm(tumor_vectors.shape[0], device=feature_map.device)[:n_tumor]
        background_index = torch.randperm(background_vectors.shape[0], device=feature_map.device)[:n_background]

        embeddings.append(tumor_vectors[tumor_index])
        embeddings.append(background_vectors[background_index])
        labels.append(torch.ones(n_tumor, device=feature_map.device, dtype=torch.long))
        labels.append(torch.zeros(n_background, device=feature_map.device, dtype=torch.long))

    if not embeddings:
        return None, None

    embeddings_tensor = torch.cat(embeddings, dim=0).view(-1, channels)
    labels_tensor = torch.cat(labels, dim=0)
    if (labels_tensor == 1).sum() < min_samples_per_class_total:
        return None, None
    if (labels_tensor == 0).sum() < min_samples_per_class_total:
        return None, None
    return embeddings_tensor, labels_tensor


def _supervised_region_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings.float(), dim=1)
    logits = embeddings @ embeddings.T
    logits = logits / temperature

    n = embeddings.shape[0]
    self_mask = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    denominator_mask = ~self_mask

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * denominator_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    positives_per_anchor = positive_mask.sum(dim=1)
    valid_anchor = positives_per_anchor > 0
    if not valid_anchor.any():
        return embeddings.new_zeros(())

    mean_log_prob_positive = (positive_mask.float() * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1)
    return -mean_log_prob_positive[valid_anchor].mean()


def _tumor_region_contrastive_loss_from_feature_map(
    feature_map: torch.Tensor,
    target: torch.Tensor | list[torch.Tensor],
    temperature: float,
    max_samples_per_class_per_case: int,
    min_samples_per_class_total: int,
    ignore_label: int,
) -> torch.Tensor:
    embeddings, labels = _sample_region_embeddings(
        feature_map,
        target,
        max_samples_per_class_per_case=max_samples_per_class_per_case,
        min_samples_per_class_total=min_samples_per_class_total,
        ignore_label=ignore_label,
    )
    if embeddings is None or labels is None:
        return feature_map.new_zeros(())
    return _supervised_region_contrastive_loss(embeddings, labels, temperature=temperature)


class PretrainedTrainer_250ep_TumorRegionContrastive(PretrainedTrainer_250ep):
    """
    MAE-65 compatible trainer with a tumor-region contrastive auxiliary loss.

    Use exactly like PretrainedTrainer_250ep, changing only the trainer name:

        -tr PretrainedTrainer_250ep_TumorRegionContrastive
    """

    contrastive_stage_index = 4
    contrastive_weight = 0.01
    contrastive_temperature = 0.1
    contrastive_max_samples_per_class_per_case = 128
    contrastive_min_samples_per_class_total = 8
    contrastive_ignore_label = -1

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        self._contrastive_activation_capture: Optional[_EncoderStageActivationCapture] = None

    def _do_i_compile(self):
        allow_compile = os.environ.get("TUMOR_CONTRASTIVE_ALLOW_COMPILE", "").lower() in ("true", "1", "t")
        if allow_compile:
            return super()._do_i_compile()
        return False

    def _ensure_activation_capture(self) -> None:
        if self._contrastive_activation_capture is None:
            self._contrastive_activation_capture = _EncoderStageActivationCapture(
                self.network,
                self.contrastive_stage_index,
            )
            self.print_to_log_file(
                "Tumor-region contrastive loss enabled: "
                f"stage={self.contrastive_stage_index}, "
                f"weight={self.contrastive_weight}, "
                f"temperature={self.contrastive_temperature}, "
                f"max_samples_per_class_per_case={self.contrastive_max_samples_per_class_per_case}, "
                f"torch_compile={self._do_i_compile()}"
            )

    def train_step(self, batch: dict) -> dict:
        self._ensure_activation_capture()

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with context:
            output = self.network(data)
            segmentation_loss = self.loss(output, target)

            feature_map = self._contrastive_activation_capture.activation
            if feature_map is None:
                raise RuntimeError(
                    "No encoder activation was captured. Check contrastive_stage_index and network.encoder.stages."
                )

            contrastive_loss = _tumor_region_contrastive_loss_from_feature_map(
                feature_map,
                target,
                temperature=self.contrastive_temperature,
                max_samples_per_class_per_case=self.contrastive_max_samples_per_class_per_case,
                min_samples_per_class_total=self.contrastive_min_samples_per_class_total,
                ignore_label=self.contrastive_ignore_label,
            )
            total_loss = segmentation_loss + self.contrastive_weight * contrastive_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "segmentation_loss": segmentation_loss.detach().cpu().numpy(),
            "contrastive_loss": contrastive_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: list[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            contrastive_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            dist.all_gather_object(seg_losses_tr, outputs["segmentation_loss"])
            dist.all_gather_object(contrastive_losses_tr, outputs["contrastive_loss"])
            loss_here = np.vstack(losses_tr).mean()
            segmentation_loss_here = np.vstack(seg_losses_tr).mean()
            contrastive_loss_here = np.vstack(contrastive_losses_tr).mean()
        else:
            loss_here = np.mean(outputs["loss"])
            segmentation_loss_here = np.mean(outputs["segmentation_loss"])
            contrastive_loss_here = np.mean(outputs["contrastive_loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)
        self.print_to_log_file(
            "train_loss_components "
            f"total={np.round(loss_here, decimals=4)} "
            f"segmentation={np.round(segmentation_loss_here, decimals=4)} "
            f"contrastive={np.round(contrastive_loss_here, decimals=4)} "
            f"weighted_contrastive={np.round(self.contrastive_weight * contrastive_loss_here, decimals=4)}"
        )


class PretrainedTrainer_250ep_TumorRegionContrastiveS4W0p03(
    PretrainedTrainer_250ep_TumorRegionContrastive
):
    contrastive_stage_index = 4
    contrastive_weight = 0.03

class PretrainedTrainer_250ep_TumorRegionContrastiveS4W0p003(
    PretrainedTrainer_250ep_TumorRegionContrastive
):
    contrastive_stage_index = 4
    contrastive_weight = 0.003


class PretrainedTrainer_250ep_TumorRegionContrastiveS4W0p000(
    PretrainedTrainer_250ep_TumorRegionContrastive
):
    contrastive_stage_index = 4
    contrastive_weight = 0.0