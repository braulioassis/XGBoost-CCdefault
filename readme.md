## Introduction
I evaluate the performance of XGBoost machine learning models in predicting credit card default risk. These models are trained on the [Default of Credit Card Clients dataset](https://doi.org/10.24432/C55S3H) maintained by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

## Methods
I build XGBoost models on python using the pandas, scikit-learn, and xgboost libraries, and I use matplotlib to visualize model performance.

## Results
### Model 1
I start with a standard XGBoost model (500 estimators, max depth of 4, learning rate of 0.05, and 80% sampling of cases and features. This model showed high overall accuracy (82%), but a low recall rate of credit card default cases (38%).
```
Classification report:
               precision    recall  f1-score   support

           0       0.84      0.94      0.89      7009
           1       0.66      0.38      0.48      1991

    accuracy                           0.82      9000
   macro avg       0.75      0.66      0.68      9000
weighted avg       0.80      0.82      0.80      9000
```
This is likely a consequence of the imbalance between no-default and default cases in the training data (~3:1). Therefore, I build a second model that includes assigned weights to the outcome variable in order to account for the data imbalance. Curiously, this model had reduced overall accuracy (76%), but a significantly higher recall rate for credit card default cases (63%).
```
Classification report:
               precision    recall  f1-score   support

           0       0.88      0.80      0.84      7009
           1       0.47      0.63      0.54      1991

    accuracy                           0.76      9000
   macro avg       0.68      0.71      0.69      9000
weighted avg       0.79      0.76      0.77      9000
```
Given that my objective is to correctly identify credit card default risk, I will favor the model 2 that has a higher recall rate for credit card default (true positives) than model 1 that has higher overall accuracy but higher false negatives.

Afterwards, I attempt to improve model 2 using hyperparameter tuning. Here, I test combinations of different number of estimators, tree depths, learning rates, and subsampling of cases and features. However, hyperparameter tuning resulted in no improvement.

Lastly, I engineer a new "utilization rate" feature, hypothesizing that customers that utilize a higher proportion of their available credit would be associated with a higher risk of default. Utilization rate was calculated as the ratio of the a month's balance and the credit limit, for each of the six months data is available for. Additionally, I created an "average utilization" feature that is the average of utilization rate across the six months. While utilization rate on the first month ranked as a third most important feature in the model, these modifications did not improve credit card default recall rate (62%).
