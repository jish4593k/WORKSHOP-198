from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tkinter import Tk, filedialog

app = Flask(__name__)

# Load articles from CSV on startup
df = pd.read_csv("articles.csv")
all_articles = df.to_dict(orient="records")

liked_articles = []
not_liked_articles = []

# Function to load regression data
def load_regression_data():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV file for regression", filetypes=[("CSV files", "*.csv")])
    if file_path:
        regression_data = pd.read_csv(file_path)
        root.destroy()
        return regression_data
    else:
        root.destroy()
        return None

# Load regression data
regression_data = load_regression_data()

# Train a simple linear regression model
if regression_data is not None:
    X = regression_data.iloc[:, :-1].values
    y = regression_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

@app.route("/get-article", methods=["GET"])
def get_article():
    if all_articles:
        return jsonify({"data": all_articles[0], "status": "success"}), 200
    else:
        return jsonify({"status": "no more articles"}), 404

@app.route("/like-dislike-article", methods=["POST"])
def like_dislike_article():
    if all_articles:
        article = all_articles.pop(0)
        action = request.json.get("action", "")
        if action == "like":
            liked_articles.append(article)
        elif action == "dislike":
            not_liked_articles.append(article)

        return jsonify({"status": "success"}), 201
    else:
        return jsonify({"status": "no more articles"}), 404

@app.route("/predict-regression", methods=["POST"])
def predict_regression():
    if regression_data is not None:
        features = request.json.get("features", [])
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": prediction, "status": "success"}), 200
    else:
        return jsonify({"status": "regression model not trained"}), 500

if __name__ == "__main__":
    app.run()
