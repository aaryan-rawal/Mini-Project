"""
app.py — Flask Web Application
Serves the spam detection UI and handles prediction requests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from app.pred import predict, get_history

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)


@app.route("/")
def index():
    history = get_history(20)
    return render_template("index.html", history=history)


@app.route("/predict", methods=["POST"])
def run_predict():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Message cannot be empty."}), 400

    msg    = data["message"].strip()
    result = predict(msg)
    return jsonify(result)


@app.route("/history")
def history():
    return jsonify(get_history(20))


if __name__ == "__main__":
    print("\n SpamShield is running!")
    print(" Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
