from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from tqdm import tqdm
import json
import re
import numpy as np
import math
from tqdm.auto import tqdm
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
)


SQUAD_MODEL = 'distilbert-base-uncased-distilled-squad'
TEST_FILE = 'test_gpt_verified.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_FILE_NAME = 'output_bert_squad_test.json'

MODEL_NAME_ENT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TOKENIZER_ENT = AutoTokenizer.from_pretrained(MODEL_NAME_ENT)
MODEL_ENT = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_ENT).to(DEVICE)


def calculate_entailment(sentence1: str, sentence2: str) -> float:

    encoding = TOKENIZER_ENT.encode_plus(
        sentence1, sentence2, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)

    outputs = MODEL_ENT(**encoding)
    logits = outputs.logits.softmax(dim=1)

    return logits[0, 0].item()


def get_macro_f1(y_true, y_pred):
    """
    Calculate macro F1 score.
    """
    
    y_true = [ord(ans)-65 for ans in y_true]
    y_pred = [ord(ans)-65 for ans in y_pred]

    return f1_score(y_true, y_pred, average='macro')


if __name__ == '__main__':
    
    with open(TEST_FILE, 'r') as file:
        data = json.load(file)

    tokenizer = DistilBertTokenizer.from_pretrained(SQUAD_MODEL)
    model = DistilBertForQuestionAnswering.from_pretrained(SQUAD_MODEL).to(DEVICE)

    results = {}
    too_long = 0
    
    # get evidence from model
    for key, value in tqdm(data.items()):
        try:
            text = key.split('<delimiter!>')[0]
            question = key.split('<delimiter!>')[1]

            inputs = tokenizer(question, text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)

            answer_start_index = torch.argmax(outputs.start_logits)
            answer_end_index = torch.argmax(outputs.end_logits)

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            output = tokenizer.decode(predict_answer_tokens)
            results[key] = f"{value}<delimiter!>{output}"
        except RuntimeError as e:
            too_long += 1
    
    # save results
    with open(RESULTS_FILE_NAME, 'w') as json_file:
        json.dump(results, json_file)
    
    with open(RESULTS_FILE_NAME, 'r') as json_file:
        output = json.load(json_file)
        
    # get correct answer based on model's evidence
    y_true, y_pred, correct = [], [], []
    ent_model = []
    
    for k, v in tqdm(output.items()):
        bert = v.split('<delimiter!>')[1]

        pattern = r'\s(?=[A-Z]\.)'
        ans_options = re.split(pattern, k.split("<delimiter!>")[2])

        answer_options = [item.strip() for item in ans_options if item.strip()]

        ans_text = next((sentence for sentence in answer_options if sentence.startswith(k.split("<delimiter!>")[3])), None)

        ent_score = calculate_entailment(ans_text, bert)

        ent_model.append(ent_score)
        ans = k.split("<delimiter!>")[3]
        
        model_answer = [calculate_entailment(item, bert) for item in candidate]
        model_answer_letter = chr(65 + e3.index(max(e3)))
        
        y_true.append(ans)
        y_pred.append(model_answer_letter)
        
        if ans == model_answer_letter:
            correct.append(1)
        else:
            correct.append(0)
    
    
    calculated_accuracy = sum(correct)/len(correct)
    calculated_f1 = get_macro_f1(y_true, y_pred)
    avg_entailment = np.mean(ent_model)
    
    print(f'Accuracy: {sum(correct)/len(correct)}')
    print(f'Macro F1 Score: {calculated_f1}')
    print(f'Entailment: {avg_entailment}')