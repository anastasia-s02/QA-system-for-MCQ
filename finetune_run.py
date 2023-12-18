from typing import Union, Any, Optional
import wandb
import os
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
from transformers import AutoTokenizer, AutoModel, EarlyStoppingCallback
from nltk.tokenize import sent_tokenize
from info_nce import InfoNCE
import json
import pandas as pd
from datasets import Dataset


BASE_MODEL_PATH = './run_main/checkpoint-5000'

TEST_DATA_PATH = './test_gpt_TOEFL_verified.json'
TRAIN_DATA_PATH = './train_gpt_TOEFL_verified.json'
SAVE_DIR_PATH = './run_tune'

TEMPERATURE_QA = 0.3
TEMPERATURE_EXP = 0.05

def load_model(path):
    """
    Load model and tokenizer.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return model, tokenizer


def get_last_hidden(output: Tensor) -> Tensor: 
    """
    Retreive las hidden layer.
    """
    return output.last_hidden_state


def average_pool(last_hidden_state: Tensor) -> Tensor: 
    """
    Average pooling.
    """
    
    return torch.mean(last_hidden_state, dim=1)


def get_sentence_spans(sent, input_text):
    """
    Map each sentence to corresponding character span in the initial text. 
    """
    
    spans = []
    current_start = 0
    for s in sent:
        ans = find_sentence(s, input_text[current_start:], current_start)
        current_start = ans[1]
        spans.append(ans)
    return spans


def find_sentence(sentence, text, skip):
    start = text.find(sentence) + skip
    end = start + len(sentence) if start != -1 else -1  # Calculate the end position
    return [start, end]


def build_char_to_token_mapping(offset_mapping: list[torch.Tensor]) -> dict[int, list[int]]:
    """
    Creates a lookup table of char -> token spans.
    """
    char_to_token_mapping = {}

    for i, offset in enumerate(offset_mapping):
        start_offset = int(offset[0].item())
        end_offset = int(offset[1].item())

        if (start_offset, end_offset) == (0, 0):
            continue

        if start_offset not in char_to_token_mapping:
            char_to_token_mapping[start_offset] = [i, -1]
        else:
            char_to_token_mapping[start_offset][0] = i

        if end_offset not in char_to_token_mapping:
            char_to_token_mapping[end_offset] = [-1, i]
        else:
            char_to_token_mapping[end_offset][1] = i
    return char_to_token_mapping


def char_span_to_token_span(
    char_span: tuple[int, int], char_to_token_mapping: dict[int, list[int]]
):
    """
    Converts chat span into token span
    """
    start, end = char_span
    if start not in char_to_token_mapping or end not in char_to_token_mapping:
        return None

    if len(char_to_token_mapping[start]) != 2 or len(char_to_token_mapping[end]) != 2:
        return None

    if char_to_token_mapping[start][0] == -1 or char_to_token_mapping[end][1] == -1:
        return None
    return char_to_token_mapping[start][0], char_to_token_mapping[end][1] + 1


def process_data_to_model_inputs(batch):
    """
    Map data to model inputs.
    """
    
    passage_and_question = [
        f"query: {passage}. Question: {question}" 
        for passage, question in zip(batch['passage'], batch['question'])
    ]

    prefix_correction = len("query: ")
    
    tokenized_input = tokenizer(
        passage_and_question,
        padding=False,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    
    batch["input_ids"] = tokenized_input.input_ids
    batch['labels'] = [1] * len(batch['passage'])
    batch["token_spans_positive"] = []
    batch["token_spans_negative"] = []
    batch["tokenized_answers_positive"] = []
    batch["tokenized_answers_negative"] = []

    for input_ids, offsets, spans_positive, spans_negative, answers_positive, answers_negative in zip(
        tokenized_input["input_ids"],
        tokenized_input["offset_mapping"],
        batch["spans_positive"],
        batch["spans_negative"],
        batch["answers_positive"],
        batch["answers_negative"]
    ):  
        positive_spans = []
        negative_spans = []
        types_positive = []
        types_negative = []

        offset_mapping = build_char_to_token_mapping(torch.Tensor(offsets))
        
        ### spans 
        for sample_type, samples in zip(["positive", "negative"], [spans_positive, spans_negative]):
            batch[f'token_spans_{sample_type}'].append([])
            for char_span in samples.split('<delimiter>'):
                char_start, char_end = char_span.split(',')
                char_start, char_end = int(char_start) + prefix_correction, int(char_end) + prefix_correction
                token_span = char_span_to_token_span((char_start, char_end), offset_mapping)
                if token_span is not None:
                    batch[f'token_spans_{sample_type}'][-1] += [token_span]
        
        ### answers 
        for sample_type, samples in zip(["positive", "negative"], [answers_positive, answers_negative]):
            batch[f'tokenized_answers_{sample_type}'].append([])
            for answer in samples.split('<delimiter>'):
                tokenized_answer = tokenizer(answer, padding=False, truncation=True, max_length=512)
                batch[f'tokenized_answers_{sample_type}'][-1] += [tokenized_answer.input_ids]
    
    del batch["spans_positive"]
    del batch["spans_negative"]
    del batch["answers_positive"]
    del batch["answers_negative"]
    del batch["passage"]
    del batch["question"]
    return batch


def strings_match(s1, s2):
    """
    Find a match between strings ignoring spaces. 
    """
    # Remove punctuation and spaces, and convert to lowercase
    s1_processed = ''.join(e.lower() for e in s1 if e.isalnum())
    s2_processed = ''.join(e.lower() for e in s2 if e.isalnum())

    # Compare the processed strings
    return s1_processed in s2_processed or s2_processed in s1_processed


def parse_data(path_to_file):
    """
    Parse data files from JSON files. 
    """
    
    data = json.load(open(path_to_file))

    passage_list, question_list, correct_answer_options_list, incorrect_answer_options_list, correct_answer_list, evidence_list = [], [], [], [], [], []

    for sample, evidence in data.items():

        passage, question, answer_options, correct_answer = sample.split('<delimiter!>')
        passage_list.append(passage)
        question_list.append(question)
        evidence_list.append(evidence)

        correct_answer_list.append(correct_answer)
        ans_options = []
        options = ['A.', 'B.', 'C.', 'D.']

        for i, option in enumerate(options):
            current_option = answer_options.split(option)[1]
            if i != 3:
                current_option = current_option.split(options[i + 1])[0]
            ans_options += [current_option.strip()]

        if correct_answer == 'A':
            result = ans_options[0]
            ans_options.pop(0)
        elif correct_answer == 'B':
            result = ans_options[1]
            ans_options.pop(1)
        elif correct_answer == 'C':
            result = ans_options[2]
            ans_options.pop(2)
        else:
            result = ans_options[3]
            ans_options.pop(3)

        correct_answer_options_list.append(result)
        incorrect_answer_options_list.append(ans_options)
    
    positive_spans_list = []
    negative_spans_list = []
    for passage, evidence in zip(passage_list, evidence_list):

        all_spans = list(zip(get_sentence_spans(sent_tokenize(passage), passage), sent_tokenize(passage)))

        positive_spans = [f'{span[0]},{span[1]}' for span, text in all_spans if strings_match(text, evidence)]
        negative_spans = [f'{span[0]},{span[1]}' for span, text in all_spans if not strings_match(text, evidence)]
        positive_spans_list.append(positive_spans)
        negative_spans_list.append(negative_spans)

    to_df = {
        'passage': passage_list, 
        'question': question_list, 
        'answers_positive': correct_answer_options_list, 
        'answers_negative': ['<delimiter>'.join(item) for item in incorrect_answer_options_list], 
        'spans_positive': ["<delimiter>".join(ps) for ps in positive_spans_list], 
        'spans_negative': ['<delimiter>'.join(ns) for ns in negative_spans_list]
    }

    df = pd.DataFrame(to_df)
    df = df[df['spans_positive'] != '']
    df = df[df['spans_negative'] != '']
    return df


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


class MCQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask)

        loss = []
        terms = 0
        for (
            embedding,
            pos_spans,
            pos_answer_input_ids,
            pos_answer_attention_mask,
            neg_spans,
            neg_answer_input_ids,
            neg_answer_attention_mask,
        ) in zip(
            outputs.last_hidden_state,
            inputs["spans_positive"],
            inputs["answers_positive"],
            inputs["answers_positive_attention_mask"],
            inputs["spans_negative"],
            inputs["answers_negative"],
            inputs["answers_negative_attention_mask"],
        ):
            pos_span_embeddings = []
            for span in pos_spans:
                pos_span_embeddings += [embedding[span[0]:span[1], :].mean(0).unsqueeze(0)]

            neg_span_embeddings = []
            for span in neg_spans:
                neg_span_embeddings += [embedding[span[0]:span[1], :].mean(0).unsqueeze(0)]

            pos_answer_embeddings = average_pool(
                get_last_hidden(
                    model(
                        pos_answer_input_ids,
                        attention_mask=pos_answer_attention_mask,
                    )
                )
            )

            neg_answer_embeddings = average_pool(
                get_last_hidden(
                    model(
                        neg_answer_input_ids,
                        attention_mask=neg_answer_attention_mask,
                    )
                )
            )
            
            if len(pos_answer_embeddings) != 0 and len(neg_answer_embeddings) != 0:
                # question-answer loss
                num_positives = pos_answer_embeddings.shape[0]
                key_answer_loss = InfoNCE(temperature=TEMPERATURE_QA, negative_mode='paired', reduction='none')(
                    embedding.mean(0).repeat(num_positives, 1), 
                    pos_answer_embeddings, 
                    neg_answer_embeddings.unsqueeze(0).repeat(num_positives, 1, 1)
                )
                loss += [key_answer_loss]

            if len(pos_span_embeddings) != 0 and len(neg_span_embeddings) != 0:
                # answer-explanation loss
                pos_span_embeddings_joined = torch.cat(pos_span_embeddings)
                neg_span_embeddings_joined = torch.cat(neg_span_embeddings)
                
                for pos_answer_embedding in pos_answer_embeddings:
                    for pos_span_embedding in pos_span_embeddings_joined:
                        loss += [InfoNCE(temperature=TEMPERATURE_EXP, negative_mode='paired', reduction='none')(
                            pos_answer_embedding.unsqueeze(0), 
                            pos_span_embedding.unsqueeze(0), 
                            neg_span_embeddings_joined.unsqueeze(0)
                        )]
        loss_reduced = torch.cat(loss).mean()
        if return_outputs:
            return loss_reduced, []
        return loss_reduced


if __name__ == '__main__':
    # parse data files
    df_test = parse_data(TEST_DATA_PATH)
    df_train = parse_data(TRAIN_DATA_PATH)

    # create datasets
    dataset_t = Dataset.from_pandas(df_train)
    dataset_e = Dataset.from_pandas(df_test)

    # load model
    model, tokenizer = load_model(BASE_MODEL_PATH)

    # map datasets
    dataset_train = dataset_t.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=100,
        num_proc=16,
    )
    dataset_eval = dataset_e.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=100,
        num_proc=16,
    )


    model, tokenizer = load_model(BASE_MODEL_PATH)
    data_collator = MCQACollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR_PATH,
        do_train=True,
        do_eval=True,
        save_steps=500,
        save_total_limit=7,
        num_train_epochs=10,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        learning_rate=1e-6,
        remove_unused_columns=False,
        ignore_data_skip=True,
        report_to=None,
        logging_steps=50,
        eval_steps=50,
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=False,
        fp16=True,
        evaluation_strategy="steps",
        label_names=['input_ids'],
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True
    )

    trainer = MCQATrainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold=1e-5, early_stopping_patience=5)]
    )


