import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Spam Detector", layout="wide")
st.title("Spam Email Classifier")
st.write("Upload test data, pick a model, and see how it performs.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

uploaded = st.file_uploader("Upload test CSV", type="csv")
choice = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.subheader("Data preview")
    st.dataframe(data.head())

    TARGET = "is_spam"
    if TARGET not in data.columns:
        st.error("The CSV must contain the 'is_spam' target column.")
    else:
        from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                                     recall_score, f1_score, matthews_corrcoef,
                                     confusion_matrix, classification_report)
        import matplotlib.pyplot as plt
        import seaborn as sns

        X_new = data.drop(columns=[TARGET])
        y_new = data[TARGET]
        model = joblib.load(MODEL_FILES[choice])
        pred = model.predict(X_new)
        prob = model.predict_proba(X_new)[:, 1]

        st.subheader("Evaluation metrics: " + choice)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", round(accuracy_score(y_new, pred), 4))
        c2.metric("AUC", round(roc_auc_score(y_new, prob), 4))
        c3.metric("MCC", round(matthews_corrcoef(y_new, pred), 4))
        c4, c5, c6 = st.columns(3)
        c4.metric("Precision", round(precision_score(y_new, pred), 4))
        c5.metric("Recall", round(recall_score(y_new, pred), 4))
        c6.metric("F1", round(f1_score(y_new, pred), 4))

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_new, pred), annot=True, fmt="d",
                    cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification report")
        st.text(classification_report(y_new, pred))
