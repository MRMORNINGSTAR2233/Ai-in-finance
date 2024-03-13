from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from genetic_algorithm_module import genetic_algorithm
from anomaly_detection_module import detect_anomalies

app = Flask(__name__)

# Configure your local SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Change 'your_database.db' to your preferred database name
db = SQLAlchemy(app)

# Define a User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

# Example API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Save the user to the local database
    new_user = User(email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify(message='User registered successfully', user={'email': email}), 201

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check if the user exists in the local database
    user = User.query.filter_by(email=email, password=password).first()

    if user:
        return jsonify(message='Login successful', user={'email': user.email}), 200
    else:
        return jsonify(error='Invalid credentials'), 401
@app.route('/api/detect-anomalies', methods=['POST'])
def api_detect_anomalies():
    data = request.json
    financial_data = np.array(data['financial_data'])  # Assuming 'financial_data' is passed in the request

    anomaly_scores, predictions = detect_anomalies(financial_data)

    return jsonify({
        'anomaly_scores': anomaly_scores.tolist(),
        'predictions': predictions.tolist()
    })

# Genetic Algorithm API endpoint
@app.route('/api/genetic-algorithm', methods=['POST'])
def api_genetic_algorithm():
    data = request.json
    expense_categories = data.get('expense_categories')
    budget_limits = data.get('budget_limits')
    financial_goals = data.get('financial_goals')

    population_size = data.get('population_size', 10)
    num_generations = data.get('num_generations', 100)
    mutation_rate = data.get('mutation_rate', 0.1)

    optimized_budget_allocation = genetic_algorithm(
        expense_categories, budget_limits, financial_goals,
        population_size=population_size, num_generations=num_generations, mutation_rate=mutation_rate
    )

    return jsonify({'optimized_budget_allocation': optimized_budget_allocation})

# Update other endpoints accordingly

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
