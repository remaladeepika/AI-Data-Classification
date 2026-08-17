from flask import Flask, render_template, request, jsonify
from classification import predict_flower, get_model_info

app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    model_info = get_model_info()
    return render_template("index.html", model_info=model_info)


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():
    model_info = get_model_info()
    return render_template("about.html", model_info=model_info)


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        sepal_length = data.get("sepal_length")
        sepal_width = data.get("sepal_width")
        petal_length = data.get("petal_length")
        petal_width = data.get("petal_width")

        if (
            sepal_length is None
            or sepal_width is None
            or petal_length is None
            or petal_width is None
        ):
            return jsonify({
                "success": False,
                "error": "Please provide all four measurements."
            }), 400

        sepal_length = float(sepal_length)
        sepal_width = float(sepal_width)
        petal_length = float(petal_length)
        petal_width = float(petal_width)

        if (
            sepal_length <= 0
            or sepal_width <= 0
            or petal_length <= 0
            or petal_width <= 0
        ):
            return jsonify({
                "success": False,
                "error": "All measurements must be greater than zero."
            }), 400

        result = predict_flower(
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        )

        return jsonify({
            "success": True,
            "prediction": result["flower"],
            "confidence": result["confidence"],
            "class_id": result["class_id"]
        })

    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please enter valid numeric values."
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MODEL INFORMATION API
# ============================================================

@app.route("/api/model-info")
def model_info():
    return jsonify(get_model_info())


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )