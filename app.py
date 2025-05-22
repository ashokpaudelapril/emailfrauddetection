# app.py
from flask import Flask, request, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"http\S+|www.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()   # remove extra spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def spam_detection(text):
    text = clean_text(text)
    text = vectorizer.transform([text])
    result = model.predict(text)
    return "Spam" if result == 1 else "Not Spam"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]
        prediction = spam_detection(email_text)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
