from flask import Flask, jsonify, request
from cnn import run_cnn

app = Flask(__name__)


@app.route("/")
def home():
    return "test"


@app.route("/image", methods=["POST"])
def image():
    file = request.files["img"]

    print(file.filename)

    file.save("temp.png")
    labels = run_cnn("temp.png")

    print("got labels", labels)
    if labels[0] == "birds":
        is_threat = False
    else:
        is_threat = True
    return jsonify({"label": labels, "isThreat": is_threat})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
