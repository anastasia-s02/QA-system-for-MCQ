# Efficient Grounded QA System for Multiple Choice Questions

In this repository we have developed an Efficient Grounded QA System for Multiple Choice Questions. Our model utilizes InfoNCE loss which significantly enhances its ability to differentiate between correct and incorrect answers. Our objective was to create accurate and intepratable model that finds the correct answer to a multiple choice question when given context, question and options. In addition to answering the question by choosing one of the multiple choice options, our model returns corroborating sentence that led it to the chosen answer.


## Dependencies
1. Python 3.6+
2. Tranformers 4.15+
3. Torch 2.0+


## Download data

**Please download the datasets (in .json format) and put them into the folder "./data". The following sections will need them.** <br>
URL: [Datasets](https://drive.google.com/drive/folders/1zXiU0i5Ma2Km0ce3TjEwyjQw2SbzA3C-?usp=sharing)


## Data transformations
To transform your raw data to the format that contrastive loss requires (key, positive answers, negative answers) you can either
- use the Dataset_Transformation.ipynb and do so mannually (recommended for those who do not have the raw data in same format as the data above), or
- you can do so by importing [data_transformation.py](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/dataset_transformations/data_transformation.py) and running transform_data_contrastive_loss("raw_file_name.json", "output_file_name").

### Example:
```bash
import Dataset_Transformations/data_transformation.py
transform_data_contrastive_loss("/data/train_TOEFL.json", "train_TOEFL_processed_contrastive_loss")
```

## Additional Repository Content Explanations 

### data_transformations Folder
This folder contains the previously mentioned notebook [Dataset_Transformation.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/dataset_transformations/Dataset_Transformation.ipynb) that can be used to mannually transform the datasets to contranstive learning format. Further it can be adapted if dataset used is not in the same raw format as ours was in (more details can be found in the notebook). 

This folder also contains [data_transformation.py](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/Dataset%20Transformations/data_transformation.py) that can be used for data transformation as listed above. 

This folder contrains a script for collecting evidence from ChatGPT (`collect_chatgpt_data.py`), which can be run by `python collect_chatgpt_data.py` after inserting personal API key for OpenAI. After data collections, it is necessary to run `verify_gpt_evidence.py` script in order to verify that ChatGPT produced evidence is indeed a part of passage. 

Our transformed RACE and TOEFL training data can be found in this folder. 

This folder contains [data_exploration_RACE.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/EDA/data_exploration_RACE.ipynb) file that shows our Exploritory Data Analysis for the RACE dataset.

### MQAG_testing Folder
This folder contains [MQAG_Testing.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/MQAG_testing/MQAG_Testing.ipynb) file that shows implementation of MQAG model on our RACE and TOEFL datasets. We can see that the model has high accuracy on both datasets in predicitng the correct answer, but it does not have the ability to corroborate the chosen answer. Due to this we do not know how the model is making a choice (black-box) thus the model lacks interpretability.

### BERT_testing Folder
The following folder contains scripts to run and evaluate three bert-based models: BERT-SWAG, BERT-SQUAD, and BERT-base. 

### testing_script_GenMC Folder
This folder contains a script to run and evaluate GenMC model. 

## Finding the best temperatures & learning rate (`training_and_eval` folder)

In order to find out which learning rate gives the best results, run `learning_rate_tuning.py` script to train `e5-base-v2` with 4 different learning rates for 1 epoch. 
After the run is complete, run `eval_lr_tuning.py` to get metrics (specifically, macro F1 score, accuracy, and entailment score). 

In order to find out which temperatures ($\tau$) give the best results, run `temperature_tuning.py` script to train `e5-base-v2` with 12 randomly samples pairs of ($\tau_{QA}$, $\tau_{exp}$) for 1 epoch. 
After the run is complete, run `eval_lemp_tuning.py` to get metrics (specifically, macro F1 score, accuracy, and entailment score).

Note: WandB API key is needed for runs to start (to be able to track training).

## Main training and fine-tuning (`training_and_eval` folder)

Run `main_training.py` to launch main training. It will terminate after stopping criterion terminates the training (patience = 5 for growing val loss) or after 12 epochs, whichever is earlier. 

Run `finetune_run.py` to launch fine-tuning (base model should be set to the latest checkpoint of resulting model after running `main_training.py`. 

Get metrics for both models on TOEFL and RACE datasets by running `eval_main_and_finetuned.py`.




