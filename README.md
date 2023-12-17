# Efficient Grounded QA System for Multiple Choice Questions

This repo...


## Dependencies
1. Python 3.6+
2. ...


## Download data

**Please download the datasets (in .json format) and put them into the folder "./data". The following sections will need them.** <br>
URL: [Datasets](https://drive.google.com/drive/folders/1zXiU0i5Ma2Km0ce3TjEwyjQw2SbzA3C-?usp=sharing)


## Data transformations
To transform your raw data to the format that contrastive loss requires (key, positive answers, negative answers) you can either
- use the Dataset_Transformation.ipynb and do so mannually (recommended for those who do not have the raw data in same format as the data above), or
- you can do so by importing [data_transformation.py](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/Dataset%20Transformations/data_transformation.py) and running transform_data_contrastive_loss("raw_file_name.json", "output_file_name").

### Example:
```bash
import Dataset_Transformations/data_transformation.py
transform_data_contrastive_loss("/data/train_TOEFL.json", "train_TOEFL_processed_contrastive_loss")
```



## Additional Repository Content Explanations 

### Data Transformations Folder
This folder contains the previously mentioned notebook [Dataset_Transformation.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/Dataset%20Transformations/Dataset_Transformation.ipynb) that can be used to mannually transform the datasets to contranstive learning format. Further it can be adapted if dataset used is not in the same raw format as ours was in (more details can be found in the notebook). 

This folder also contains [data_transformation.py](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/Dataset%20Transformations/data_transformation.py) that can be used for data transformation as listed above. 

Additionally, our transformed RACE and TOEFL training data can be found in this folder. 


### EDA Folder
This folder contains [data_exploration_RACE.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/EDA/data_exploration_RACE.ipynb) file that shows our Exploritory Data Analysis for the RACE dataset.

### MQAG_testing Folder
This folder contains [MQAG_Testing.ipynb](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/MQAG_testing/MQAG_Testing.ipynb) file that shows implementation of MQAG model on our RACE and TOEFL datasets. We can see that the model has high accuracy on both datasets in predicitng the correct answer, but it does not have the ability to corroborate the chosen answer. Due to this we do not know how the model is making a choice (black-box) thus the model lacks interpretability.


### testing_script_GenMC Folder

