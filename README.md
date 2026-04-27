# 🛡 SpamShield — AI-Powered SMS Spam Detector

A complete machine learning project that detects spam messages using **Naive Bayes**, **Logistic Regression**, and **Linear SVM** trained on the UCI SMS Spam Collection dataset.

> Built with Python · Flask · scikit-learn · NLTK · TF-IDF

---

## 📁 Project Structure

```
spam_project/
│
├── data/
│   └── spam.csv               ← Download and place here
│
├── ml/
│   ├── prep.py                ← Text preprocessing pipeline
│   ├── train.py               ← Model training + comparison
│   └── eval.py                ← Evaluation + confusion matrix
│
├── app/
│   ├── app.py                 ← Flask web application
│   └── pred.py                ← Prediction engine + history logger
│
├── model/
│   ├── model.pkl              ← Auto-generated after training
│   └── vec.pkl                ← Auto-generated after training
│
├── templates/
│   └── index.html             ← Main web UI
│
├── static/
│   ├── style.css              ← Styling
│   └── script.js              ← Frontend JS
│
├── history.csv                ← Auto-generated on first prediction
├── confusion_matrix.png       ← Auto-generated after eval
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Download the Dataset

- Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Download and extract the zip
- Place `spam.csv` inside the `data/` folder

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Order

Run all commands from inside the `spam_project/` folder:

```bash
# Step 1 — Train the model
python -m ml.train

# Step 2 — Evaluate the model
python -m ml.eval

# Step 3 — Launch the web app
python -m app.app
```

Then open: **http://127.0.0.1:5000**

---

## 📊 Expected Results

| Model | Typical Accuracy |
|-------|-----------------|
| Naive Bayes | ~97–98% |
| Logistic Regression | ~98–99% |
| Linear SVM | ~98–99% |

---

## 👥 Team

- Project: SpamShield
- Supervisor: Mr. Rick Chatterjee
- University: GLA University

---

## 📄 License

MIT License — free to use for academic and personal projects.
