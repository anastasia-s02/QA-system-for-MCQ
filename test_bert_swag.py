from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch
from tqdm import tqdm
import json
import re


SWAG_MODEL = "gueleilo/bert-base-uncased-finetuned-swag"
TEST_FILE = 'test_gpt_verified.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_FILE_NAME = 'output_bert_swag_test.json'
    
    
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
    
    tokenizer = AutoTokenizer.from_pretrained(SWAG_MODEL)
    model = AutoModelForMultipleChoice.from_pretrained(SWAG_MODEL).to(DEVICE)

    pattern = r'\s(?=[A-Z]\.)'

    results = {}
    too_long = 0

    for key, value in tqdm(data.items()):
        try:
            prompt = f"CONTEXT: {key.split('<delimiter!>')[0]} QUESTION: {key.split('<delimiter!>')[1]}"

            items = re.split(pattern, key.split("<delimiter!>")[2])

            candidate = [item.strip() for item in items if item.strip()]


            inputs = tokenizer([[prompt, candidate[i]] for i in range(len(candidate))], return_tensors="pt", padding=True)
            labels = torch.tensor(0).unsqueeze(0)

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
            logits = outputs.logits

            predicted_class = logits.argmax().item()
            
            new_key = f"{key}<delimiter!>{chr(65+predicted_class)}"
            results[new_key] = value
        except RuntimeError as e:
            too_long += 1
            
    with open(RESULTS_FILE_NAME, 'w') as json_file:
        json.dump(results, json_file)
    
    with open(RESULTS_FILE_NAME, 'r') as json_file:
        output = json.load(json_file)
        
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for k in output:
        ans = k.split("<delimiter!>")[4]
        res = k.split("<delimiter!>")[5]
        y_true.append(ans)
        y_pred.append(res)
        total +=1
        if ans == res:
            correct += 1
    
    calculated_f1 = get_macro_f1(y_true, y_pred)
    
    print(f'Accuracy: {correct/total}')
    print(f'Macro F1 Score: {calculated_f1}')