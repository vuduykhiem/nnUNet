
import torch
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer

class PretrainedTrainer_250ep(PretrainedTrainer):

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
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15 # lin increase whole network
        self.num_epochs = 250

class PretrainedTrainer_500ep(PretrainedTrainer):

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
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 50 # lin increase whole network
        self.num_epochs = 500