# Code base of the project **Explainable Bengali Multiclass News Classification**

This repository contains the code for the project Explainable Bengali Multiclass News Classification!

Before running the code, we need to install specific libraries. We provided conda environment.

```terminal
conda env create -f environment.yml
conda activate explainable-news
```
## Data pre-processing

For pre-processing the data follow the steps in `data_preprocess.ipynb` notebook file. After pre-processing, the final data is saved in the `data` directory.


## Training & Testing

For training the model run the following:

```terminal
python3 train.py
```

Before testing the model, we need to load the saved weights from the `saved_weights` folder. Copy the saved weight's file name and paste it in line 38 of `test.py` 

```python
saved_model_path = f'saved_weights/<weight file name>'
```

Now, for testing, run the follwoing:

```terminal
python3 test.py
```

Running the test file will give the test accuracy for the model.


## Explainability

Follow the `Explainability.ipynb` notebook. Load the saved weights like the testing process, in cell 4 of the notebook.


## Reference

1. Dataset: https://www.kaggle.com/datasets/furcifer/bangla-newspaper-dataset
2. Bangla Bert: https://huggingface.co/csebuetnlp/banglabert_large
3. Explainability library: https://captum.ai/ 