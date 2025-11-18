
import numpy as np
from flask import Flask, jsonify
from autarkeia.engine import AutarkeiaMetrics

app = Flask(__name__)


@app.route("/")
def index():
    return {
        "message": "Autarkeia Metrics Checker demo",
        "endpoints": ["/api/demo"],
    }


@app.route("/api/demo")
def api_demo():
    # toy demo with random embeddings
    series = [np.random.randn(256, 64) for _ in range(5)]
    m = AutarkeiaMetrics.from_embeddings(series)
    res = m.run()

    return jsonify(
        {
            "T": res.meta.get("T"),
            "rank": res.meta.get("rank"),
            "completion_mean": float(res.completion.mean()),
            "completion_std": float(res.completion.std()),
            "hysteresis": float(res.hysteresis),
            "eoi": float(res.eoi),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8008, debug=True)
