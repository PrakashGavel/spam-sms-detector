# Spam SMS Detector

A Flask-based web application that classifies SMS messages as Spam or Not Spam using Machine Learning.

## Features
- TF-IDF + Logistic Regression spam classifier
- Web UI using Flask
- 96% accuracy on test data
- Trained on spam_ham_dataset.csv

## Tech Stack
- Python
- Flask
- scikit-learn
- NLTK

## How to Run

```bash
git clone <repo-url>
cd spam-sms-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
