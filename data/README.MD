Download the dataset from Kaggle, and palce them in this folder.

## Dataset from kaggle

- `train.csv`
- `test.csv`

## Pre-processed dataset

- `raw_train.npz`: training data extracted from `train.csv` and saved in the numpy formation
    + `data`
    + `label`
- `raw_test.npz`: test data extracted from `test.csv` and saved in the numpy formation
    + `data`
- `splited_train.npz`: split the raw training data into training dataset and validation dataset
    + `train_data`
    + `train_label`
    + `validate_data`
    + `validate_label`
- `augmented_train.npz`: the dataset for the splited training dataset
    + `data`
    + `label`
