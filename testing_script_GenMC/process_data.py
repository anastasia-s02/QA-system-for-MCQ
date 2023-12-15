import json

with open('QA_data/test_gpt_verified.json','r') as f:
  raw_test = json.load(f)
  f.close()

with open('QA_data/train_gpt15k_verified.json','r') as f:
  raw_train = json.load(f)
  f.close()

#define a funfction to organize data
def process_data(raw,name,sep = '<delimiter!>'):
  id = 0
  with open(name, 'w') as file:
    choices = ['A. ','B. ','C. ','D. ']
    for k in raw.keys():
      l = k.split(sep)
      ch = l[2]
      choice_list = []
      for i in range(1,len(choices)):
        text = ch[ch.find(choices[i-1]) + 3 : ch.find(choices[i])]
        choice_list.append({'text':text, 'label': choices[i-1][0]})
      text = ch[ch.find(choices[i]) + 3:]
      choice_list.append({'text':text, 'label': choices[i][0]})
      line = {'id': '%s'%id,
              'question': {'stem':l[0] + l[1],
                          'choices':choice_list},
              'answerKey':l[3],
              'explanation': raw[k]}
      json_string = json.dumps(line)
      file.write(json_string + '\n')
      id += 1

    file.close()

process_data(raw_train, 'train_data_QA.jsonl')
process_data(raw_test,'test_data_QA.jsonl')


