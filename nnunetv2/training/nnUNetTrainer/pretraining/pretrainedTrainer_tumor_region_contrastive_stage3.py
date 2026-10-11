"""
Stage-3 tumor-vs-background contrastive trainer.

This is a controlled follow-up to
PretrainedTrainer_250ep_TumorRegionContrastive: it keeps the same auxiliary
loss, weight, temperature, sampling, MAE loading, plans, and fold handling, but
captures encoder stage 3 instead of stage 4.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer_tumor_region_contrastive import (
    PretrainedTrainer_250ep_TumorRegionContrastive,
)


class PretrainedTrainer_250ep_TumorRegionContrastiveStage3(PretrainedTrainer_250ep_TumorRegionContrastive):
    """MAE-65 compatible tumor-region contrastive trainer using encoder stage 3."""

    contrastive_stage_index = 3
