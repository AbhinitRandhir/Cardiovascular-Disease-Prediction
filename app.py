from flask import Flask, render_template, request, redirect, session
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# -----------------------------
# Load ML Model
# -----------------------------

model = joblib.load("ml/models/random_forest.pkl")
scaler = joblib.load("ml/processed/scaler.pkl")

# Features used by model
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# -----------------------------
# User database
# -----------------------------

USER_DB = "users.json"

if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)


def load_users():
    with open(USER_DB, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=4)


# -----------------------------
# Home
# -----------------------------

@app.route("/")
def home():

    if "user" not in session:
        return redirect("/login")

    return render_template(
        "index.html",
        name=session["name"]
    )


# -----------------------------
# Signup
# -----------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        email = request.form["email"]
        password = request.form["password"]

        users = load_users()

        if email in users:
            return "User already exists!"

        users[email] = {
            "name": name,
            "age": age,
            "gender": gender,
            "password": password
        }

        save_users(users)

        session["user"] = email
        session["name"] = name

        return redirect("/")

    return render_template("signup.html")


# -----------------------------
# Login
# -----------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        users = load_users()

        if email in users and users[email]["password"] == password:

            session["user"] = email
            session["name"] = users[email]["name"]

            return redirect("/")

        return "Invalid email or password"

    return render_template("login.html")


# -----------------------------
# Logout
# -----------------------------

@app.route("/logout")
def logout():

    session.clear()
    return redirect("/login")


# -----------------------------
# Predict Page
# -----------------------------

@app.route("/predict")
def predict_page():

    if "user" not in session:
        return redirect("/login")

    return render_template("predict.html")


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    try:

        values = []

        for f in FEATURE_NAMES:
            values.append(float(request.form[f]))

        values = np.array(values).reshape(1, -1)

        scaled = scaler.transform(values)

        prediction = model.predict(scaled)[0]

        prob = model.predict_proba(scaled)[0][1]

        if prediction == 1:
            risk = "High Risk"
            image = "high_risk.png"

            foods = [
                "Oatmeal",
                "Fatty fish (Salmon, Tuna)",
                "Avocado",
                "Walnuts and Almonds",
                "Green vegetables (Spinach, Broccoli)",
                "Olive oil",
                "Garlic",
                "Berries (Strawberry, Blueberry)",
                "Beans and Lentils",
                "Whole grains"
            ]

        else:
            risk = "Low Risk"
            image = "low_risk.png"

            foods = [
                "Fresh fruits (Apple, Banana, Orange)",
                "Whole grains (Oats, Brown rice, Whole wheat)",
                "Nuts (Almonds, Walnuts)",
                "Leafy vegetables (Spinach, Lettuce)",
                "Carrots and Tomatoes",
                "Olive oil",
                "Low-fat dairy products",
                "Coconut water",
                "Green tea"
            ]

        return render_template(
            "predict.html",
            prediction=risk,
            probability=round(prob * 100, 2),
            image=image,
            foods=foods
        )

    except Exception as e:
        return f"Prediction error: {str(e)}"


# -----------------------------
# About
# -----------------------------

@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------
# Run App
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)