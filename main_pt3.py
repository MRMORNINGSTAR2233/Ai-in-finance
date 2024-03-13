from flask import Flask, request, jsonify
from supabase import create_client
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

app = Flask(__name__)

# Supabase Configuration (update with your Supabase config)
supabase_url = 'your_supabase_url'
supabase_key = 'your_supabase_key'
supabase = create_client(supabase_url, supabase_key)

# Example API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Use Supabase authentication API instead of Firebase
    user = supabase.auth.sign_up(email=email, password=password)

    return jsonify(message='User registered successfully', user=user), 201

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Use Supabase authentication API instead of Firebase
    user = supabase.auth.sign_in(email=email, password=password)

    return jsonify(message='Login successful', user=user), 200


# Anomaly Detection API endpoint
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

# Add more API endpoints for other functionalities

if __name__ == '__main__':
    app.run(debug=True)
