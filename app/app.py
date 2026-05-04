"""
app.py — Flask Web Application  (SpamShield v2)
────────────────────────────────────────────────
New endpoints over v1:
  GET  /api/model-info   →  live model metadata (accuracy, trained_at, source)
  POST /api/refresh      →  re-fetch dataset from network + retrain models
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from app.pred import predict, get_history, reload_model, get_model_info

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__),   "..", "static"),
)

_retrain_lock = threading.Lock()
_retrain_status = {"running": False, "last_result": None}


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


@app.route("/api/model-info")
def model_info():
    """Return live model training metadata."""
    return jsonify(get_model_info())


@app.route("/api/refresh", methods=["POST"])
def refresh():
    """
    Re-fetch dataset from the web and retrain.
    Runs in a background thread; returns immediately with status 202.
    Poll /api/model-info to see when training completes.
    """
    if _retrain_status["running"]:
        return jsonify({"status": "already_running",
                        "message": "Retraining already in progress."}), 409

    def _retrain():
        _retrain_status["running"] = True
        try:
            from ml.train import run as train_run
            acc = train_run(force_remote=True)
            meta = reload_model()
            _retrain_status["last_result"] = {
                "status":   "success",
                "accuracy": acc,
                "meta":     meta,
            }
        except Exception as e:
            _retrain_status["last_result"] = {"status": "error", "error": str(e)}
        finally:
            _retrain_status["running"] = False

    t = threading.Thread(target=_retrain, daemon=True)
    t.start()

    return jsonify({
        "status":  "started",
        "message": "Retraining started. Dataset will be fetched from network.",
    }), 202


@app.route("/api/refresh-status")
def refresh_status():
    return jsonify({
        "running":     _retrain_status["running"],
        "last_result": _retrain_status["last_result"],
    })


if __name__ == "__main__":
    print("\n SpamShield v2 is running!")
    print(" Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
