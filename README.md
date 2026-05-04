spamshield.zip
Zip Archive
For this project give me the latest readme

Here’s a clean, updated, professional README (v3-ready) you can directly replace your current one with:

🚀 SpamShield v3 — Intelligent Spam Detection System
B.Tech Project | GLA University, Mathura (2025–26)
Team: Aaryan Rawal, Aditya Kumar, Aryan Prakash, Ram Prakash Pandey

🔥 Overview
SpamShield is an NLP-powered spam detection web application that classifies messages as Spam or Ham using machine learning.
It combines advanced preprocessing, dynamic dataset fetching, and real-time prediction APIs into a full-stack deployable system.

✨ Key Features
🧠 Intelligent NLP Pipeline
POS-aware lemmatization (context-based word normalization)

Semantic token handling (__url__, __phone__, __money__)

Spam-sensitive stopword filtering (keeps words like free, win, urgent)

📊 Machine Learning
TF-IDF Vectorization

Trained classification model (saved as model.pkl)

Confidence score + keyword extraction

🌐 Dynamic Dataset System
Auto-fetch dataset from internet (UCI / GitHub)

Local caching for performance

Fallback mechanism (no crashes if offline)

🔁 Live Model Retraining
/api/refresh → retrain model without restarting server

Background processing (async)

📈 Model Transparency
/api/model-info endpoint

Training metadata:

Accuracy

Dataset source

Timestamp

💻 Web Interface
Clean UI with:

Input box for prediction

Result with confidence

NLP feature breakdown

Model info banner

📂 Project Structure
spamshield/
│
├── app/
│   ├── app.py          # Flask backend + routes
│   ├── pred.py         # Prediction logic
│
├── ml/
│   ├── train.py        # Model training pipeline
│   ├── prep.py         # NLP preprocessing
│   ├── data_fetcher.py # Dataset loader (dynamic)
│   ├── eval.py         # Evaluation metrics
│
├── data/
│   ├── spam.csv        # Local fallback dataset
│   ├── spam_cache.csv  # Cached dataset
│
├── model/
│   ├── model.pkl       # Trained model
│   ├── vec.pkl         # TF-IDF vectorizer
│   ├── train_meta.json # Training metadata
│
├── templates/
│   └── index.html      # Frontend UI
│
├── static/
│   ├── style.css
│   └── script.js
│
├── requirements.txt
└── README.md
⚙️ Installation & Setup
# Clone project
git clone <your-repo-link>
cd spamshield

# Install dependencies
pip install -r requirements.txt

# Train model
python ml/train.py

# Run server
python app/app.py
👉 Open: http://127.0.0.1:5000

📡 API Endpoints
Method	Endpoint	Description
GET	/	Web interface
POST	/predict	Predict spam/ham
GET	/history	Prediction history
GET	/api/model-info	Model metadata
POST	/api/refresh	Retrain model
GET	/api/refresh-status	Check retrain status
🧪 NLP Pipeline (Simplified)
Lowercase text

Replace patterns:

URLs → __url__

Phone → __phone__

Money → __money__

Tokenization (NLTK)

Stopword removal (spam-safe)

POS tagging

Lemmatization

🌍 Dataset Handling Logic
Priority order:

Cached dataset (<24 hrs)

UCI Dataset (online fetch)

GitHub backup dataset

Local CSV fallback

📊 Sample Output
{
  "prediction": "spam",
  "confidence": 0.97,
  "keywords": ["free", "win", "__url__"],
  "model_info": {
    "accuracy": 0.982,
    "trained_at": "2026-05-01"
  }
}
🚀 Future Improvements
Deep Learning (LSTM / BERT)

Multilingual spam detection

Email + SMS integration

Deployment on cloud (AWS / Render)

User authentication & dashboards

🏁 Conclusion
SpamShield demonstrates a complete ML lifecycle system:

Data → Processing → Training → API → UI → Retraining

It is production-ready in architecture and suitable for real-world deployment with further scaling.

If you want, I can next:

Make this GitHub-optimized (badges + screenshots)

OR convert it into a project report / synopsis (for viva)
