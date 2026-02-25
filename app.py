from flask import Flask, render_template, request
import os
from train import train

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    images = sorted(os.listdir("static/generated"))
    metrics_img = None

    if request.method == "POST":
        epochs = int(request.form["epochs"])
        train(epochs)
        images = sorted(os.listdir("static/generated"))
        metrics_img = "metrics.png"

    return render_template("index.html", images=images, metrics_img=metrics_img)


if __name__ == "__main__":
    app.run(debug=True)