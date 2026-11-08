## Spam Email Classifier

## a. Problem statement
 - The goal is to decide whether an email is spam (1) or not spam (0) using the frequency of certain words and characters in the message. I trained five classification models on the same data and built a Streamlit app that evaluates each of them on an uploaded test file.

## b. Dataset description
- The dataset is Spambase from the UCI Machine Learning Repository. It has 4,601 emails with 57 numeric features (word frequencies, character frequencies, and capital-letter run lengths) and a binary target column, `is_spam`. After removing 391 duplicate rows, 4,210 rows remained. I split them 80/20 with stratification, which gave 3,368 training rows and 842 test rows. The test rows are saved in `test_data.csv`.

## c. GitHub repository link
 - https://github.com/deepam1212/spam-email-classifier

## d. Models used

All five models were trained on the same 3,368 training rows and evaluated on the same 842 test rows. Logistic Regression, KNN and SVM use a StandardScaler inside a pipeline.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9347 | 0.9768 | 0.9297 | 0.9048 | 0.9170 | 0.8634 |
| Decision Tree | 0.9074 | 0.9053 | 0.8817 | 0.8869 | 0.8843 | 0.8071 |
| KNN | 0.9133 | 0.9546 | 0.9148 | 0.8631 | 0.8882 | 0.8184 |
| Naive Bayes | 0.8361 | 0.9525 | 0.7240 | 0.9524 | 0.8226 | 0.6975 |
| Support Vector Machine | 0.9323 | 0.9772 | 0.9319 | 0.8958 | 0.9135 | 0.8584 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy, F1 and MCC overall. Balanced precision and recall, so it works well on this mostly linear word-frequency data. |
| Decision Tree | Weakest AUC (0.9053) and low MCC. A single tree likely overfits the training data and does not generalise as well. |
| KNN | Good precision (0.9148) but lower recall (0.8631), so it misses more spam than the top models. |
| Naive Bayes | Highest recall (0.9524) but lowest precision (0.7240), so it flags many normal emails as spam. Its independence assumption does not hold for correlated word counts. |
| Support Vector Machine | Almost tied with Logistic Regression, with the highest AUC (0.9772) and precision (0.9319), but slightly lower recall. |
| **Overall winner** | Logistic Regression, because it leads on accuracy, F1 and MCC. SVM is a very close second. |
