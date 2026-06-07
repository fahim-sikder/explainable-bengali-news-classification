# Explainable Bengali Multiclass News Classification

This repository contains the code for **Explainable Bengali Multiclass News Classification** — fine-tuning [BanglaBERT (large)](https://huggingface.co/csebuetnlp/banglabert_large) to classify Bengali news articles into 9 categories, and interpreting the model's predictions with [Captum](https://captum.ai/).

## Setup

The dependencies are provided as a conda environment:

```terminal
conda env create -f environment.yml
conda activate explainable-news
```

The code runs on both GPU and CPU (CUDA is used automatically when available).

## Project structure

| File | Description |
| --- | --- |
| `data_preprocess.ipynb` | Cleans the raw dataset and writes the processed CSV to the `data/` directory. |
| `data-stats.ipynb` | Exploratory statistics of the dataset (e.g. class balance, see `imbalanced.png`). |
| `utils.py` | Model definition (`ClassifyNews`) and the `NewsDatasets` PyTorch dataset. |
| `train.py` | Trains the classifier and saves weights to `saved_weights/`. |
| `test.py` | Loads saved weights and reports the test classification report. |
| `Explainability.ipynb` | Attributes predictions to input tokens using Captum. |
| `label_encoder.pkl` | Fitted label encoder mapping class names to indices. |

> **Note:** `data/` and `saved_weights/` are not tracked in this repository (they are large). Generate the data with `data_preprocess.ipynb` and the weights with `train.py`.

## Data pre-processing

Follow the steps in `data_preprocess.ipynb`. After pre-processing, the final data is saved in the `data/` directory as `even_cleaned_data.csv` (a class-balanced version of the cleaned dataset).

## Training & testing

Both training and testing use the **same** dataset (`data/even_cleaned_data.csv`) and the **same** stratified split (`test_size=0.2`, `random_state=2023`). Training uses the 80% partition and testing uses the disjoint 20% partition, so there is no train/test leakage.

To train the model:

```terminal
python3 train.py
```

Trained weights are written to `saved_weights/`. Before testing, set the weight file you want to evaluate at line 38 of `test.py`:

```python
saved_model_path = f'saved_weights/<weight file name>'
```

Then run:

```terminal
python3 test.py
```

This prints the per-class precision/recall/F1 and overall accuracy on the held-out test set.

## Explainability

Follow `Explainability.ipynb`. Load the saved weights the same way as in testing (cell 4 of the notebook), then run the Captum-based attribution cells to see which tokens drive each prediction.

## References

1. Dataset: https://www.kaggle.com/datasets/furcifer/bangla-newspaper-dataset
2. BanglaBERT: https://huggingface.co/csebuetnlp/banglabert_large
3. Explainability library: https://captum.ai/

## Citation

If you use this repository, please cite our paper:

> DOI: [10.1109/ICCIT60459.2023.10441218](https://doi.org/10.1109/ICCIT60459.2023.10441218)

```bibtex
@inproceedings{sikder2023explainable,
    title={Explainable Bengali Multiclass News Classification},
    author={Sikder, Md Fahim and Ferdous, Md and Afroz, Shraboni and Podder, Uzzal and Fatema, Kaniz and Hossain, Mohammad Nahid and Hasan, Md Tahmid and Baowaly, Mrinal Kanti},
    booktitle={2023 26th International Conference on Computer and Information Technology (ICCIT)},
    pages={1--6},
    year={2023},
    organization={IEEE}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
