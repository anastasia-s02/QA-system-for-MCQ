# Efficient Grounded QA System for Multiple Choice Questions

This repo...


## Dependencies
1. Python 3.6+
2. ...


## Download data

**Please download the datasets (in .json format) and put them into the folder "./data". The following sections will need them.** <br>
URL: [Datasets](https://drive.google.com/drive/folders/15NI1k_np1-ZCgIdZzAkwI9xO8Zqsg4C0)


## Data transformations
To transform your raw data to the format that contrastive loss requires (key, positive answers, negative answers) you can either
- use the Dataset_Transformation.ipynb and do so mannually (recommended for those who do not have the raw data in same format as the data above), or
- you can do so by importing [data_transformation.py](https://github.com/anastasia-s02/QA-system-for-MCQ/blob/main/Dataset%20Transformations/data_transformation.py) and running transform_data_contrastive_loss("raw_file_name.json", "output_file_name").

### Example:
```bash
import Dataset_Transformations/data_transformation.py
transform_data_contrastive_loss("/data/train_TOEFL.json", "train_TOEFL_processed_contrastive_loss")
```
