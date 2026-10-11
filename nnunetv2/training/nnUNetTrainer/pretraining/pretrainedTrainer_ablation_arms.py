"""
Ablation arms B/C/D trainer subclasses (ablation_B_C_prompt.md, Phase 1).

MECHANISM DECISION (this session): B/C/D run OFFLINE, matching how arm D's real in-thesis
result was actually produced -- the Dataset193-197 staging pipeline (docs/
SYNTHETIC_LYTIC_DATASET_PIPELINE.md), NOT a per-training-step online transform. D bakes one
fixed synthetic lesion into ~85% of training cases, once, before training starts; B and C
must be delivered the same way or "D beats C" is confounded by online-vs-offline delivery,
not by the thing under test. The online carve-transform + seg-channel-hiding architecture
this file originally scaffolded (get_training_transforms override, StripTransform, per-worker
RNG) is DEFERRED -- that's a separate, later "online + difficulty curriculum" experiment, not
part of validating the current offline result. Arm C is built by reusing
prepare_from_manifest_65.py -> synth_verify_staging_65.py -> synth_assemble_folds_65.py
unchanged, with only the Step-0 generator swapped (naive region + isotropic sphere +
bootstrap-inflation radius instead of v20's GT-anchored/compact-elongated recipe). See
scripts/ablation_arm_c_naive65_generator.py.

FAIRNESS RULE: B, C, D differ ONLY in their tested axis. All three subclass
PretrainedTrainer_250ep directly (not PretrainedTrainer) so architecture, schedule,
epochs, and fold handling are identical BY INHERITANCE -- not by three configs that
happen to agree today and drift tomorrow. If PretrainedTrainer_250ep is the wrong base
(e.g. the ablation is meant to run a different epoch budget than the 250ep production
schedule), change it here once, at the top, for all three arms simultaneously -- do not
give one arm a different base class than the others.

Arm B (this file, ArmB_LyticOversamplingTrainer): NO synthetic lesion, and (explicit user
decision) delivered via nnUNetDataLoader's native `sampling_probabilities` rather than
physical case duplication in a staged dataset. These are the PROVABLY IDENTICAL distribution
-- nnUNetDataLoader draws every case via np.random.choice(indices, ..., p=probs) with
replacement from a fixed pool regardless of whether that pool holds one weighted entry or
many uniform-weight duplicates. Physical duplication would only matter if it changed the
per-draw CONTENT (as D's staging does, baking in a new lesion instance) -- B has no generated
content to bake in, so there is nothing for "offline" to buy here beyond dataset footprint.
Chose to keep this tested subclass over an 18-32x-duplicated fold dataset for the same result.
Writeup note: "arm B uses weighted sampling; arms C/D use baked datasets; for B these are
provably equivalent since B adds no generated content."

Target rates are arm D's REAL, MEASURED per-fold augmented-training-case rate (from the v20
full-65 production manifest x splits_final_65.json, confirmed this session) -- NOT the 0.33
online-insertion-rate figure used earlier, which was wrong for this offline comparison.

Arm D subclass: not needed as a trainer class at all under the offline decision -- D is
just PretrainedTrainer_250ep trained on Dataset193-197 directly, already built.
"""
import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer_variants import PretrainedTrainer_250ep
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

# Resolved and verified this session from case_manifest.csv + patient_tracking.csv:
# order N == OrthoTUMOR_{N:03d}, cross-checked against all 12 IDs by name (see conversation).
# Do not regenerate this list from "order" numbers without that same cross-check --
# the mapping is NOT a trivial assumption, it happens to hold, and was confirmed as such.
LYTIC_CASE_IDS = frozenset({
    "OrthoTUMOR_006", "OrthoTUMOR_012", "OrthoTUMOR_014", "OrthoTUMOR_022",
    "OrthoTUMOR_027", "OrthoTUMOR_033", "OrthoTUMOR_039", "OrthoTUMOR_044",
    "OrthoTUMOR_050", "OrthoTUMOR_054", "OrthoTUMOR_061", "OrthoTUMOR_065",
})

# Arm D's ACTUAL per-fold augmented-training-case rate, measured from
# outputs/_smoketest_v20_full65/_progress_manifest.csv (55/65 augmented) crossed against
# splits_final_65.json's per-fold train lists. Fold index -> rate.
PER_FOLD_LYTIC_SAMPLING_RATE = {
    0: 0.846,   # 44/52
    1: 0.846,   # 44/52
    2: 0.885,   # 46/52
    3: 0.808,   # 42/52
    4: 0.846,   # 44/52
}


class ArmB_LyticOversamplingTrainer(PretrainedTrainer_250ep):
    """
    Arm B: real-lytic case oversampling via sampling_probabilities, no synthetic lesions.
    """

    def _build_lytic_sampling_probabilities(self, identifiers) -> np.ndarray:
        """
        Per-case sampling-probability array so the real-lytic cases are drawn at a
        combined rate matching arm D's ACTUAL per-fold augmented-case rate
        (PER_FOLD_LYTIC_SAMPLING_RATE), not a fixed constant -- D's rate itself varies
        slightly fold-to-fold (0.808-0.885) because placement success/failure isn't
        evenly distributed across the cohort. self.fold selects the right target.

        POSITIONAL: probs[i] is the draw probability of identifiers[i]. Verified this
        session (see conversation) that identifiers is sorted exactly once, in
        nnUNetBaseDataset.__init__, and never reordered/replaced afterward -- including
        during training, since nnUNetDataLoader runs with infinite=True, which skips the
        only shuffle-on-reset path that exists in the base DataLoader class. Safe to build
        this once at get_dataloaders time and trust the positions for the whole run.

        identifiers : dataset_tr.identifiers for THIS fold (already sorted).
        returns     : np.ndarray shape (len(identifiers),), dtype float64, sums to 1.0.
        """
        identifiers = list(identifiers)
        n = len(identifiers)
        target_rate = PER_FOLD_LYTIC_SAMPLING_RATE[self.fold]

        is_lytic = np.array([cid in LYTIC_CASE_IDS for cid in identifiers], dtype=bool)
        L = int(is_lytic.sum())        # lytic cases in THIS fold's train list
        n_other = n - L

        # --- regression guard: catches sampling from the WRONG list (val leaked in, wrong ID set) ---
        # Expected per-fold train-lytic counts (T=52): fold0=8, fold1=9, fold2=10, fold3=10, fold4=11.
        assert L in {8, 9, 10, 11}, (
            f"Lytic count L={L} not in expected {{8,9,10,11}} -- identifiers is not the fold "
            f"train list this rule was built against. n={n}, "
            f"matched={sorted(c for c in identifiers if c in LYTIC_CASE_IDS)}"
        )
        assert n_other > 0, "no non-lytic cases to oversample against -- arm B framing broken"

        probs = np.empty(n, dtype=np.float64)
        probs[is_lytic] = target_rate / L                 # split across the L lytic cases
        probs[~is_lytic] = (1.0 - target_rate) / n_other  # split across the rest

        # np.random.choice requires p to sum to exactly 1 (within fp tolerance)
        assert np.isclose(probs.sum(), 1.0), f"probs sum {probs.sum()} != 1"
        return probs

    def get_dataloaders(self):
        """
        Copied from nnUNetTrainer.get_dataloaders (nnUNetTrainer.py:640-709) with exactly
        one change: dl_tr gets a computed sampling_probabilities array instead of None.
        dl_val (below) is built from the SAME dataset_val / val_transforms the base class
        would use, completely untouched -- confirm this stays true if you ever need to
        touch this method again; the fairness of arm B depends on val never being reweighted.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # --- the one real change vs. the base class ---
        lytic_sampling_probabilities = self._build_lytic_sampling_probabilities(dataset_tr.identifiers)

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                  initial_patch_size, self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=lytic_sampling_probabilities,
                                  pad_sides=None, transforms=tr_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                   self.configuration_manager.patch_size, self.configuration_manager.patch_size,
                                   self.label_manager,
                                   oversample_foreground_percent=self.oversample_foreground_percent,
                                   sampling_probabilities=None,  # unchanged -- val is never reweighted
                                   pad_sides=None, transforms=val_transforms,
                                   probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
