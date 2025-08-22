## Spatial Uncertainty Quantification in DeepKriging: A Bayesian Approach

This repository was developed as part of my Master’s thesis at Humboldt-Universität zu Berlin.

It implements a probabilistic deep learning pipeline to predict PM2.5 concentrations across the United States. The model builds on Bayesian deep learning techniques to quantify predictive uncertainty.

### Repository structure

```text
.
├── data/                  # Input dataset and data preprocessing
├── results/               # Saved predictions for the full dataset
├── save_models/           # Saved weights from the best performing epoch per fold
├── utils/                 # Core model components
│   ├── bayesian_model.py       # Bayesian DeepKriging model
│   ├── training.py             # Model training
│   ├── prediction.py           # Prediction with 200 MC samples
│   ├── evaluation.py           # Evaluation metrics (RMSE, CRPS, NLL)
│   ├── visualization.py        # For visualization PM2.5 concentration and uncertainty
│   └── __init__.py
├── config.py              # Hyperparameter configuration
├── DNN_baseline.py        # Baseline deterministic model
├── main.ipynb             # Jupyter notebook for experiments and plots
├── environment.yml
├── requirements.txt
└── README.md
```
