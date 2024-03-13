from flask import Flask, request, jsonify
from firebase import Firebase
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

# Firebase Configuration
firebase_config = {
    'apiKey': 'your_api_key',
    'authDomain': 'your_auth_domain',
    'databaseURL': 'your_database_url',
    'projectId': 'your_project_id',
    'storageBucket': 'your_storage_bucket',
    'messagingSenderId': 'your_messaging_sender_id',
    'appId': 'your_app_id',
}

firebase = Firebase(firebase_config)
auth = firebase.auth()
db = firebase.database()

# Example API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    try:
        user = auth.create_user_with_email_and_password(email, password)
        return jsonify(message='User registered successfully', user=user), 201
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return jsonify(message='Login successful', user=user), 200
    except Exception as e:
        return jsonify(error=str(e)), 401

@app.route('/api/financial-advice', methods=['POST'])
def get_financial_advice():
    data = request.json
    user_id = data.get('user_id')

    # Fetch user's financial data from Firebase Realtime Database
    user_data = db.child('users').child(user_id).get().val()

    # Implement your financial analysis and advice logic here

    advice = {'message': 'Financial advice generated based on user data'}
    return jsonify(advice), 200

# ... (Additional AI and financial endpoints as needed)

# Function to detect anomalies using Isolation Forest
def detect_anomalies(data, contamination=0.05):
    """
    Parameters:
    - data: 2D numpy array or pandas DataFrame, where each row represents a data point.
    - contamination: The proportion of outliers in the data (default is 0.05).

    Returns:
    - anomaly_scores: An array of anomaly scores for each data point.
    - predictions: Binary values indicating whether each data point is an anomaly (1) or not (0).
    """

    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)

    # Fit the model to the data and obtain anomaly scores
    anomaly_scores = model.fit_predict(data)

    # Convert the anomaly scores to binary predictions
    predictions = np.where(anomaly_scores == -1, 1, 0)

    return anomaly_scores, predictions

# Example usage
if __name__ == "__main__":
    # Assuming 'financial_data' is a 2D array or DataFrame with financial transaction data
    # Each row represents a transaction, and each column represents a feature (e.g., amount, timestamp, etc.)
    financial_data = np.array([[...], [...], ...])

    # Detect anomalies in the financial data
    anomaly_scores, predictions = detect_anomalies(financial_data)

    # Print or use the anomaly scores and predictions as needed
    print("Anomaly Scores:", anomaly_scores)
    print("Predictions:", predictions)

# Initialize population for genetic algorithm
def initialize_population(population_size, expense_categories, budget_limits):
    population = []
    for _ in range(population_size):
        budget_allocation = {category: np.random.uniform(0, limit) for category, limit in budget_limits.items()}
        population.append(budget_allocation)
    return population

# Calculate fitness for genetic algorithm
def calculate_fitness(budget_allocation, financial_goals):
    fitness = sum(abs(budget_allocation[category] - financial_goals[category]) for category in budget_allocation)
    return fitness

# Genetic algorithm for budget optimization
def genetic_algorithm(expense_categories, budget_limits, financial_goals, population_size=10, num_generations=100, mutation_rate=0.1):
    population = initialize_population(population_size, expense_categories, budget_limits)

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual, financial_goals) for individual in population]
        parents_indices = np.argsort(fitness_scores)[:population_size // 2]
        parents = [population[i] for i in parents_indices]

        offspring = []
        for _ in range(population_size - len(parents)):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            crossover_point = np.random.choice(len(expense_categories))
            child = {category: parent1[category] if np.random.rand() < 0.5 else parent2[category] for category in expense_categories}
            offspring.append(child)

        for child in offspring:
            if np.random.rand() < mutation_rate:
                category_to_mutate = np.random.choice(expense_categories)
                child[category_to_mutate] = np.random.uniform(0, budget_limits[category_to_mutate])

        population = parents + offspring

    final_fitness_scores = [calculate_fitness(individual, financial_goals) for individual in population]
    best_allocation = population[np.argmin(final_fitness_scores)]

    return best_allocation

# Get user inputs for genetic algorithm
def get_user_inputs():
    expense_categories = input("Enter expense categories separated by commas: ").split(',')
    expense_categories = [category.strip() for category in expense_categories]

    budget_limits = {}
    for category in expense_categories:
        limit = float(input(f"Enter budget limit for {category}: $"))
        budget_limits[category] = limit

    financial_goals = {}
    for category in expense_categories:
        goal = float(input(f"Enter financial goal for {category}: $"))
        financial_goals[category] = goal

    return expense_categories, budget_limits, financial_goals

# Main function for genetic algorithm
def main_genetic_algorithm():
    expense_categories, budget_limits, financial_goals = get_user_inputs()

    population_size = int(input("Enter population size: "))
    num_generations = int(input("Enter number of generations: "))
    mutation_rate = float(input("Enter mutation rate (between 0 and 1): "))

    optimized_budget_allocation = genetic_algorithm(
        expense_categories, budget_limits, financial_goals,
        population_size=population_size, num_generations=num_generations, mutation_rate=mutation_rate
    )

    print("\nOptimized Budget Allocation:")
    for category, amount in optimized_budget_allocation.items():
        print(f"{category}: ${amount:.2f}")

# Example usage of genetic algorithm
if __name__ == "__main__":
    main_genetic_algorithm()

# Load and preprocess data for ensemble modeling
data = pd.read_csv('your_dataset.csv')  # Update with the actual path to your dataset

X = data.drop('category', axis=1)
y = data['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
logistic_model = LogisticRegression()
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# Create an ensemble model (voting classifier)
ensemble_model = VotingClassifier(estimators=[
    ('logistic', logistic_model),
    ('tree', tree_model),
    ('forest', forest_model)
], voting='hard')

ensemble_model.fit(X_train, y_train)
ensemble_predictions = ensemble_model.predict(X_test)

# Evaluate the models
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_model.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_model.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, forest_model.predict(X_test)))
print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))

# You can also print classification reports for more detailed metrics
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logistic_model.predict(X_test)))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, tree_model.predict(X_test)))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, forest_model.predict(X_test)))
print("\nEnsemble Model Classification Report:\n", classification_report(y_test, ensemble_predictions))

# Debt payoff optimizer using dynamic programming
def debt_payoff_optimizer(debts):
    """
    Debt payoff algorithm using dynamic programming.

    Parameters:
    debts (list): List of dictionaries representing debts.
                  Each dictionary should have keys 'balance', 'interest_rate', and 'min_payment'.

    Returns:
    int: Minimum total payments to achieve debt freedom.
    """

    # Sort debts by interest rate in descending order
    debts.sort(key=lambda x: x['interest_rate'], reverse=True)

    num_debts = len(debts)
    num_months = len(debts[0]['balance'])

    # Initialize a 2D array to store minimum payments for each debt and month
    dp = [[float('inf')] * (num_debts + 1) for _ in range(num_months + 1)]

    # Base case: No debts remaining, so the total payment is zero
    for i in range(len(dp[0])):
        dp[0][i] = 0

    # Dynamic programming iteration
    for i in range(1, num_debts + 1):
        for j in range(1, num_months + 1):
            # Calculate the minimum payment to pay off debt i in month j
            min_payment = min(dp[i - 1][j], debts[i - 1]['min_payment'] + dp[i][j - 1])

            # Update the dp table with the minimum payment
            dp[i][j] = min_payment + (debts[i - 1]['balance'][j - 1] * debts[i - 1]['interest_rate'] / 12)

    # The minimum total payment to achieve debt freedom is stored in the bottom-right cell
    return int(dp[num_debts][num_months])

# Example usage:
debts = []

num_debts = int(input("Enter the number of debts: "))

for i in range(num_debts):
    balance = list(map(int, input(f"Enter the balance for debt {i + 1} separated by spaces: ").split()))
    interest_rate = float(input(f"Enter the interest rate for debt {i + 1} (as a decimal): "))
    min_payment = int(input(f"Enter the minimum monthly payment for debt {i + 1}: "))

    debt_info = {'balance': balance, 'interest_rate': interest_rate, 'min_payment': min_payment}
    debts.append(debt_info)

minimum_total_payment = debt_payoff_optimizer(debts)
print(f"The minimum total payment to achieve debt freedom is: ${minimum_total_payment}")

# Time series forecasting using ARIMA
def evaluate_arima_model(train_data, test_data, order):
    history = list(train_data)
    predictions = []

    # Walk-forward validation
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test_data[t])

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_data, predictions))

    return rmse, predictions

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define the order of the ARIMA model (p, d, q)
order = (5, 1, 0)  # Example values, you may need to fine-tune

# Evaluate the ARIMA model
rmse, predictions = evaluate_arima_model(train, test, order)

# Print RMSE
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(test, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('ARIMA Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()

# Portfolio optimization using mean-variance optimization
def calculate_portfolio_statistics(weights, returns, cov_matrix):
    portfolio_return = np.sum(weights * returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def objective_function(weights, returns, cov_matrix, target_return):
    portfolio_return, portfolio_volatility = calculate_portfolio_statistics(weights, returns, cov_matrix)
    # Minimize the negative of the Sharpe ratio to maximize the Sharpe ratio
    return -(portfolio_return - target_return) / portfolio_volatility

def optimize_portfolio(returns, cov_matrix, target_return, risk_tolerance=1.0):
    num_assets = len(returns)

    # Define the constraints for the optimization
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Define the bounds for the weights (each weight should be between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initialize weights equally
    initial_weights = np.ones(num_assets) / num_assets

    # Optimize the portfolio using the scipy minimize function
    result = minimize(
        objective_function,
        initial_weights,
        args=(returns, cov_matrix, target_return),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    optimized_weights = result.x

    # Adjust weights based on risk tolerance
    adjusted_weights = risk_tolerance * optimized_weights / np.sum(optimized_weights)

    return adjusted_weights

# Example usage:
# Assuming you have historical returns and covariance matrix available
historical_returns = np.array([0.05, 0.08, 0.12, 0.10])
covariance_matrix = np.array([[0.001, 0.0005, 0.001, 0.0008],
                              [0.0005, 0.002, 0.001, 0.001],
                              [0.001, 0.001, 0.003, 0.002],
                              [0.0008, 0.001, 0.002, 0.002]])

# Target return for the portfolio
target_return = 0.1

# Risk tolerance parameter (adjust as needed)
risk_tolerance = 1.0

# Optimize the portfolio
optimized_weights = optimize_portfolio(historical_returns, covariance_matrix, target_return, risk_tolerance)

# Print the optimized weights for each asset
print("Optimized Weights:")
for i, weight in enumerate(optimized_weights):
    print(f"Asset {i + 1}: {weight:.4f}")

# K-means clustering for customer segmentation
def k_means_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

# Example usage:
# Assuming 'customer_data' is a DataFrame with customer features
# Adjust 'num_clusters' based on the desired number of segments
num_clusters = 3
customer_segments = k_means_clustering(customer_data, num_clusters)

# Add the cluster labels to the original DataFrame
customer_data['Cluster'] = customer_segments

# Print the resulting customer segments
print("Customer Segments:")
for cluster in range(num_clusters):
    segment_data = customer_data[customer_data['Cluster'] == cluster]
    print(f"Segment {cluster + 1}:\n", segment_data)
