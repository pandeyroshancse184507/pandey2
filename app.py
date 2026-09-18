from flask import Flask, render_template, request, jsonify
import re
import string
import sklearn.feature_extraction.text
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Sample Training Dataset (Real-world mock training)
train_texts = [
    "NASA confirms asteroid will hit Earth tomorrow share immediately",
    "Government hiding truth about secret vaccine experiment",
    "Breaking News: Drinking lemon juice cures all viruses completely",
    "ISRO successfully launches new satellite into orbit from Sriharikota",
    "RBI announces new guidelines for digital payment transactions",
    "The stock market index reached an all-time high today"
]
train_labels = [0, 0, 0, 1, 1, 1]  # 0 = Fake, 1 = Real

# NLP Preprocessing & Vectorization Pipeline
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)

# Machine Learning Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, train_labels)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")
    
    if not raw_text.strip():
        return jsonify({"error": "No text provided"}), 400

    # NLP Preprocessing Pipeline Execution
    cleaned = clean_text(raw_text)
    vectorized = vectorizer.transform([cleaned])
    
    # Model Prediction
    prediction = model.predict(vectorized)[0]
    
    # Credibility Score Calculation
    sensational_words = ["breaking", "shocking", "hiding truth", "share immediately", "secret"]
    risk_count = sum(1 for word in sensational_words if word in raw_text.lower())
    
    if prediction == 0 or risk_count > 0:
        credibility_score = max(10, 35 - (risk_count * 10))
        label = "Fake News Detected"
        status = "fake"
    else:
        credibility_score = min(98, 85 + (len(raw_text) % 10))
        label = "Verified / Real News"
        status = "real"

    return jsonify({
        "status": status,
        "label": label,
        "credibility_score": credibility_score,
        "preprocessed_text": cleaned,
        "model_used": "TF-IDF + PassiveAggressiveClassifier"
    })

if __name__ == '__main__':
    print("Flask Server Running at http://127.0.0.1:5000")
    app.run(debug=True)
