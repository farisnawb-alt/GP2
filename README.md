# COVID-19 Vaccine Sentiment Analysis (Twitter/X)

Graduation project (team of 4), Umm Al-Qura University, Data Science.

## Overview
NLP pipeline that classifies the sentiment of COVID-19 vaccine tweets.
Trained on 50,000 human-labeled tweets (Sentiment140) to avoid circular
labeling, then applied to 399,595 vaccine-related tweets (63.8% positive).

## Method
- Text cleaning and TF-IDF features (5,000 uni/bi-grams)
- Logistic Regression and Naive Bayes
- 5-fold cross-validation and temporal validation

## Results
- Accuracy: 76.1% (5-fold CV, ±0.002)
- Weighted F1 on temporal split: 0.72

## Files
- `Faris_Ali_Sentiment_Project_Fixed.ipynb`: analysis and training
- `app.py`: interactive web app (Streamlit)
- `sentiment_model.joblib`, `tfidf_vectorizer.joblib`: trained model and vectorizer
- `requirements.txt`: dependencies

## Run the app
```
pip install -r requirements.txt
streamlit run app.py
```
