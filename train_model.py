import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam_ham_dataset.csv", engine="python", on_bad_lines="skip")

# Keep only needed columns
df = df[['text', 'label_num']]
df = df.rename(columns={'text': 'message', 'label_num': 'label'})

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned'] = df['message'].astype(str).apply(clean_text)

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorization with n-grams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("\n Model Evaluation")
print("----------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n Model and vectorizer saved successfully!")
