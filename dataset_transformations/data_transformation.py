import numpy as np
import json
import re

def process_file_full(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    processed_data = []
    for idx, (key, value) in enumerate(data.items(), start=1):
        parts = key.split('<delimiter!>')
        #print(parts)
        context = parts[0].strip()
        question = parts[1].strip()

        # Splitting answer choices using regular expressions
        answer_choices_str = parts[2].strip()
        answer_choices = re.split(r'\s(?=[A-D]\.)', answer_choices_str)
        answer_choices = [choice.strip() for choice in answer_choices]

        correct_answer = parts[3].strip()

        processed_data.append({
            'id': idx,
            'context': context,
            'question': question,
            'answer_choices': answer_choices,
            'correct_answer': correct_answer,
            'explanation': value
        })

    return processed_data


def transform_data_full(raw_data, output_file_name):
    processed_data = process_file_full(raw_data)

    output=[]

    for item in processed_data:
        id = item['id']
        context = item['context'].replace('\n', ' ')
        question = item['question']
        answer_choices = item['answer_choices']
        correct_answer = item['correct_answer']
        explanation = item['explanation']

        answer_choices_string = ' '.join(item['answer_choices'])

        formatted_string = f"{id}<delimiter!>{context}<delimiter!>{question}<delimiter!>{answer_choices_string}<delimiter!>{correct_answer}<delimiter!>{explanation}"
        output.append(formatted_string)

        #print(id) # to see what step we are on

    with open(f"{output_file_name}.json", 'w') as file:
        json.dump(output, file)




def process_file_contrastive_loss(file_path):
    letter_to_num = {"A":0, "B":1, "C":2, "D":3} # to convert letter to answer

    with open(file_path, 'r') as file:
        data = json.load(file)

    processed_data = []
    for idx, (key, value) in enumerate(data.items(), start=1):
        parts = key.split('<delimiter!>')
        context = parts[0].strip()
        question = parts[1].strip()

        # Splitting answer choices using regular expressions
        answer_choices_str = parts[2].strip()
        answer_choices = re.split(r'\s(?=[A-D]\.)', answer_choices_str)
        answer_choices = [choice.strip() for choice in answer_choices]

        correct_answer = parts[3].strip()

        positive_answer = answer_choices[letter_to_num[correct_answer]]

        negative_answer = [x for x in answer_choices if x != positive_answer]

        processed_data.append({
            'key': context + ":" + question,
            'positive_answers': positive_answer+":"+value,
            'negative_answers': negative_answer
        })


    return processed_data



def transform_data_contrastive_loss(raw_data, output_file_name):
    processed_data = process_file_contrastive_loss(raw_data)

    output=[]

    for item in processed_data:
        key = item['key']
        positive_answers = item['positive_answers']
        negative_answers = ' '.join(item['negative_answers'])

        formatted_string = f"{key}<delimiter!>{positive_answers}<delimiter!>{negative_answers}"
        output.append(formatted_string)

        #print(id) # to see what step we are on

    with open(f"{output_file_name}.json", 'w') as file:
        json.dump(output, file)


## SAMPLE USAGE
        
#from data_transformation import *
## If you want to process the data for direct use in python 
#processed_data_full = process_file_full("train_TOEFL_verified.json")
#processed_data_contrastive= process_file_contrastive_loss("train_TOEFL_verified.json")

## If you want to process the data and save the output automatically
#transform_data_full("train_TOEFL_verified.json", "train_TOEFL_processed_full")
#transform_data_contrastive_loss("train_TOEFL_verified.json", "train_TOEFL_processed_contrastive_loss")

