## Spam Email Classifier

## a. Problem statement
 - The goal is to decide whether an email is spam (1) or not spam (0) using the frequency of certain words and characters in the message. I trained five classification models on the same data and built a Streamlit app that evaluates each of them on an uploaded test file.

## b. Dataset description
- The dataset is Spambase from the UCI Machine Learning Repository. It has 4,601 emails with 57 numeric features (word frequencies, character frequencies, and capital-letter run lengths) and a binary target column, `is_spam`. After removing 391 duplicate rows, 4,210 rows remained. I split them 80/20 with stratification, which gave 3,368 training rows and 842 test rows. The test rows are saved in `test_data.csv`.

## c. GitHub repository link
 - https://github.com/deepam1212/spam-email-classifier
