import pandas as pd
import torch
from transformers import trainer, TrainingArguments, TrainerState, TrainerControl
from transformers import TrainerCallback
import wandb

class LengthCallback(TrainerCallback):
    def __init__(self, val_data: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_data = val_data

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global_step = state.global_step
        with torch.no_grad():
            model = self.model