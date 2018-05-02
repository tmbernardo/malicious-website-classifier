# Malicious and Benign Website Classifier with XGBoost

### Requirements
* Conda: [conda.io](https://conda.io/docs/user-guide/install/index.html) or `$ brew install caskroom/cask/anaconda`

### Usage
* Create Environment: `$ conda env create -f environment.yml`
* Activate Environment: `$ source activate benign_malicious_clf`
* Run Classifier Notebook: `$ jupyter notebook malben_clf.ipynb`

## XGBoost Parameters

### Booster Parameters
* eta = 0.18
* min_child_weight = 3
* max_depth = 6
* gamma = 0.2
* subsample = 0.75
* colsample_bytree = 0.75
* lambda = 1
* alpha = 0.05
