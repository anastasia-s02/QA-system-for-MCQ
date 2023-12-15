from sklearn.metrics import f1_score
import numpy as np
import json

enc = {'A':0, 'B':1, 'C':2, 'D':3}
predicted = []
with open('lr_5e-05_seed_1_bs_8_ga_1_layer_num_1_alpha_1.0_beta_0.5/dev.csv','r') as file:
    for line in file:
        predicted.append(enc.get(line[-2],-1))

    file.close()

true = []
with open('test_data_QA.jsonl','r') as file:
    for line in file:
        data = json.loads(line)
        true.append(enc[data['answerKey']])
    file.close()

predicted = np.array(predicted)
true = np.array(true)

print('acc: %s'%np.mean(predicted == true))
print('F1: %s'%f1_score(true,predicted,average = 'macro'))





    

