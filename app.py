from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        sms = request.form["sms"]
        cleaned = clean_text(sms)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prediction = "🚫 Spam" if pred == 1 else "✅ Not Spam"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
