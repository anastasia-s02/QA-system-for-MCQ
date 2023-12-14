from typing import Union, Any, Optional

import datasets
import torch
from torch.nn import functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
import torch
import nltk
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from info_nce import InfoNCE
import json
import pandas as pd
from datasets import Dataset


class MCQACollator(DataCollatorWithPadding):
    """
    This collator only applies padding to the passage/question inputs and generates attention masks.
    Spans for positive / negative samples are converted to tensors and kept intact.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, Any] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert list of inputs into padded batch
        """
        batch = {}
        padded_feature = self.tokenizer.pad(
            {"input_ids": [item["input_ids"] for item in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["input_ids"] = padded_feature["input_ids"]
        batch["attention_mask"] = padded_feature["attention_mask"]

        batch["spans_positive"] = []
        batch["spans_negative"] = []
        batch["answers_positive"] = []
        batch["answers_negative"] = []
        batch["answers_negative_attention_mask"] = []
        batch["answers_positive_attention_mask"] = []
        for sample in features:
            batch["spans_positive"].append(torch.LongTensor(sample["token_spans_positive"]))
            batch["spans_negative"].append(torch.LongTensor(sample["token_spans_negative"]))
            
            answers_positive = self.tokenizer.pad(
                {"input_ids": sample["tokenized_answers_positive"]},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            if "attention_mask" in answers_positive:
                batch["answers_positive"] += [answers_positive["input_ids"]]
                batch["answers_positive_attention_mask"] += [answers_positive["attention_mask"]]
            else:
                batch["answers_positive"] += [torch.Tensor([])]
                batch["answers_positive_attention_mask"] += [torch.Tensor([])]

            types_negative = self.tokenizer.pad(
                {"input_ids": sample["tokenized_answers_negative"]},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            if "attention_mask" in types_negative:
                batch["answers_negative"] += [types_negative["input_ids"]]
                batch["answers_negative_attention_mask"] += [types_negative["attention_mask"]]
            else:
                batch["answers_negative"] += [torch.Tensor([])]
                batch["answers_negative_attention_mask"] += [torch.Tensor([])]
        return batch