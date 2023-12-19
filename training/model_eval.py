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
from tqdm import tqdm


MODEL_PATH = ''
TEST_FILE_PATH = ''

    
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


def get_repr_of_span(last_hidden_state: Tensor, start, end) -> Tensor:
    """
    Slice tensor.
    """
    return last_hidden_state[:, start:end, :]


def get_cos(tensor1, tensor2):
    """
    Get cosine similarity between two tensors. 
    """
    
    return F.cosine_similarity(tensor1, tensor2).item()


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


def get_passage_question_repr(passage, question):
    """
    Get representation of passage and question.
    """
    
    batch_dict = tokenizer(f'query:{passage}{question}', max_length=512, padding=True, truncation=True, return_tensors='pt', return_offsets_mapping=True)
    outputs = model(input_ids = batch_dict['input_ids'],token_type_ids=batch_dict['token_type_ids'],
                attention_mask = batch_dict['attention_mask'])
    embeddings = average_pool(get_last_hidden(outputs))
    return embeddings, batch_dict['offset_mapping'][0], outputs


def get_answers_repr(ans_options):
    """
    Get representation for each answer option. 
    """
    ans_emb = []
    for opt in ans_options:
        batch_dict = tokenizer(f'passage:{opt}', max_length=512, padding=True, truncation=True, return_tensors='pt', return_offsets_mapping=True)
        outputs = model(input_ids = batch_dict['input_ids'],token_type_ids=batch_dict['token_type_ids'],
                attention_mask = batch_dict['attention_mask'])
        ans_emb.append(average_pool(get_last_hidden(outputs)))
    return ans_emb


def get_passage_question_extended_repr(input_text, repr, offsets, outputs):
    """
    Split test into sentences and get representation of each sentence. 
    """
    
    split_sent = sent_tokenize(input_text)
    sent_spans = get_sentence_spans(split_sent, input_text)
    token_map = build_char_to_token_mapping(offsets)

    complete_token_map = []
    
    for i in sent_spans:
        ans = char_span_to_token_span(i, token_map)
        if ans: 
            complete_token_map.append(ans)
    sentence_repr = []

    for i in complete_token_map:
        sentence_repr.append(average_pool(get_repr_of_span(get_last_hidden(outputs), i[0], i[1])))

    return split_sent, sentence_repr


def get_macro_f1(y_true, y_pred):
    """
    Calculate macro F1 score.
    """
    
    y_true = [ord(ans)-65 for ans in y_true]
    y_pred = [ord(ans)-65 for ans in y_pred]

    return f1_score(y_true, y_pred, average='macro')


def get_accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    """
    
    total = 0
    correct = 0
    for y, y1 in zip(y_true, y_pred):
        if y == y1:
            correct +=1
        total +=1
    return correct / total


if __name__ == '__main__':
    
    model, tokenizer = load_model(MODEL_PATH)
    
    # load test file
    with open(TEST_FILE_PATH, 'r') as file:
        data = json.load(file)
        

    # parse test file
    passage_l, question_l, correct_answer_l, ans_options_l = [], [], [], []

    for k, v in data.items():
        passage, question, answer_options, correct_answer = k.split('<delimiter!>')
        passage_l.append(passage)
        question_l.append(question)
        correct_answer_l.append(correct_answer)

        ans_options = []
        options = ['A. ', 'B. ', 'C. ', 'D. ']

        for i, option in enumerate(options):
            current_option = answer_options.split(option)[1]
            if i != 3:
                current_option = current_option.split(options[i + 1])[0]
            ans_options += [current_option.strip()]
        ans_options_l.append(ans_options)
        
    model_answer_l = []
    model_evidence_l = []
    
    # run model for each item in test file
    for passage, quesion, correct_answer, ans_options in tqdm(zip(passage_l, question_l, correct_answer_l, ans_options_l)):

        key_embedding, offsets, outputs = get_passage_question_repr(passage, question)
        ans_emb = get_answers_repr(ans_options)
        cos_dist = []
        for a in ans_emb:
            cos_dist.append(get_cos(a, key_embedding))

        right_ans_tensor = ans_emb[cos_dist.index(max(cos_dist))]
        model_answer_l.append(chr(cos_dist.index(max(cos_dist))+65))

        split_sent, sentence_repr = get_passage_question_extended_repr(f'query:{passage}{question}', key_embedding, offsets, outputs)

        cos_dist_all = []
        for a in sentence_repr:
            cos_dist_all.append(get_cos(a, right_ans_tensor))
        

        
        for c, sent in zip(cos_dist_all, split_sent):
            if c == max(cos_dist_all):
                model_evidence_l.append(sent)
                break
    
    # calculate metrics
    calculated_accuracy = get_accuracy(correct_answer_l, model_answer_l)
    calculated_f1 = get_macro_f1(correct_answer_l, model_answer_l)
    
    print(f'Accuracy:{calculated_accuracy}')
    print(f'Macro F1 Score:{calculated_f1}')