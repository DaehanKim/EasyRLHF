"""
stacks : huggingface trainer, huggingface datasets, deepspeed
"""


from transformers.trainer import Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import GPT2ForSequenceClassification
from transformers import GPT2Tokenizer
from transformers import TrainerCallback
from transformers.trainer_utils import (
    get_last_checkpoint,
    set_seed,
)
from transformers.utils import (
    logging,
)
from transformers.trainer_pt_utils import (
    nested_detach,
)
from torch import inf

from dataclasses import dataclass, field
from typing import Optional
import logging
import transformers
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, set_seed, AutoConfig
import os
import pickle
import torch
import sys
import numpy as np

from typing import List, Tuple, Dict, Optional, Union, Any
import torch.nn as nn


logger = transformers.logging.get_logger()


class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # Here model is GPT2ForSequenceClassification that maps a sequence of tokens to a scalar value
        # input keys : win_ids, win_mask, lose_ids, lose_mask
        # we optimize `win_logit - lose_logit` to be log-odds so the label is always 1

        win = model(
            input_ids=inputs["win_ids"],
            attention_mask=inputs["win_mask"],
        )
        win_logit, win_emb = win.logits, win.hidden_states

        lose = model(
            input_ids=inputs["lose_ids"],
            attention_mask=inputs["lose_mask"],
        )
        lose_logit, lose_emb = lose.logits, lose.hidden_states
        labels = torch.ones_like(lose_logit, dtype=torch.float)

        # loss = -torch.nn.LogSigmoid()(win_logit - lose_logit).mean()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            win_logit - lose_logit, labels
        )  # replace with binary_cross_entropy_with_logits
        return (loss, win_emb, lose_emb) if return_outputs else loss

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: Optional[str] = None
    ):
        # OVERRIDE : disable removing columns from dataset
        logger.info(
            "disabling removing unused columns from dataset object -> processed in collator!"
        )
        return dataset

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, *outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            embeddings = outputs

        if prediction_loss_only:
            return (loss, None, None)

        embeddings = nested_detach(embeddings)
        if len(embeddings) == 1:
            embeddings = embeddings[0]

        return (loss, embeddings, labels)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="microsoft/phi-1_5",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    specify max_len
    """

    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "json files containing the train data. can be comma-seperated values of json files"
        },
    )
    valid_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "json files containing the validation data. can be comma-seperated values of json files"
        },
    )

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("a data file is required.")


@dataclass
class DataCollatorForRMTraining:
    """
    modify input names from transformers DataCollatorWithPadding
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 4
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        win = [{"input_ids": e["win"]} for e in features]
        lose = [{"input_ids": e["lose"]} for e in features]
        pos_batch = self.tokenizer.pad(
            win,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        neg_batch = self.tokenizer.pad(
            lose,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return {
            "win_ids": pos_batch["input_ids"],
            "win_mask": (pos_batch["input_ids"] != self.tokenizer.pad_token_id).long(),
            "lose_ids": neg_batch["input_ids"],
            "lose_mask": (neg_batch["input_ids"] != self.tokenizer.pad_token_id).long(),
        }


def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, TrainingArguments, ModelArguments)
    )
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    transformers.logging.get_logger("transformers.configuration_utils").setLevel(
        logging.ERROR
    )
    transformers.logging.get_logger("transformers.modeling_utils").setLevel(
        logging.ERROR
    )
    transformers.logging.get_logger("transformers.tokenization_utils_base").setLevel(
        logging.ERROR
    )

    # Log on each process the small summary:
    if training_args.local_rank == 0:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"data args {data_args}")

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    def data_tokenize(example):
        # gpt2 tokenizer doesn't automatically add <|endoftext|> token at the end of text.
        # so we manually add this eos token. This is to be used as `end-of-context-token` in Preference Model finetuning
        # refer to paper `https://arxiv.org/abs/2204.05862`
        win = tokenizer(
            example["chosen"] + "<|endoftext|>", return_tensors="pt", return_length=True
        )
        example["win"] = win.input_ids[0]
        example["win_length"] = win.length.item()
        lose = tokenizer(
            example["rejected"] + "<|endoftext|>",
            return_tensors="pt",
            return_length=True,
        )
        example["lose"] = lose.input_ids[0]
        example["lose_length"] = lose.length.item()
        return example

    model_name = model_args.model_name_or_path
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # add pad token for gpt2 model - it is not used anyway due to attention mask but collator needs this intermediate steps
    # tokenizer.add_special_tokens({'additional_special_tokens':["[PAD]"]})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    reward_model = GPT2ForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    )  # returns a scalar

    with training_args.main_process_first():
        all_data = load_dataset(
            "json",
            data_files={
                "train": data_args.train_file.split(","),
                "valid": data_args.valid_file.split(","),
            },
        )
        all_data = all_data.map(data_tokenize, num_proc=10)
        # filter out sequences that is > max_seq_length
        all_data = all_data.filter(
            lambda e: e["win_length"] < data_args.max_seq_length
            and e["lose_length"] < data_args.max_seq_length,
            num_proc = 5
        )

    reward_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )  # reward model config.pad_token_id set
    train_data, valid_data = all_data["train"], all_data["valid"]
    trainer = RMTrainer(
        model=reward_model,
        args=training_args,
        data_collator=DataCollatorForRMTraining(
            tokenizer, max_length=data_args.max_seq_length, padding='max_length'
        ),
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
