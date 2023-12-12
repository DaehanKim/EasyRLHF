'''
Brought and simplified from
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
'''
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments

from trl import SFTTrainer, is_xpu_available

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="open-orca/slimorca-dedup", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

def main():
    tqdm.pandas()

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load the model

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        trust_remote_code=script_args.trust_remote_code
    )

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, split="train")

    # SFTTrainer requires a text field which model uses as target
    # if your dataset is in cascaded format like slimorca, add your manual preprocessing here. 
    if script_args.dataset_name == "open-orca/slimorca-dedup":
        from utils import process_slimorca
        with training_args.main_process_first():
            dataset = dataset.map(process_slimorca, batch_size=1, num_proc=10) # reformat text field
            dataset = dataset.filter(lambda e : len(e['text'].strip()) != 0, num_proc=5) # filter non


    # Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.max_seq_length,
        train_dataset=dataset,
        dataset_text_field=script_args.dataset_text_field,
        neftune_noise_alpha = training_args.neftune_noise_alpha)

    trainer.train()
    # Save the model
    trainer.save_model(training_args.output_dir)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
