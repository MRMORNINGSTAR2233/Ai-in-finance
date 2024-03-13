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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from anomaly_detection_module import detect_anomalies


app = Flask(__name__)

# Firebase Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your User model
class User(db.Model):
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String, nullable=False)
    password = db.Column(db.String, nullable=False)

# Example API endpoint
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Create a new user and add it to the database
    try:
        new_user = User(id=str(np.random.randint(1000000, 9999999)), email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify(message='User registered successfully', user={'id': new_user.id, 'email': new_user.email}), 201
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check if the user exists and the password is correct
    try:
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            return jsonify(message='Login successful', user={'id': user.id, 'email': user.email}), 200
        else:
            return jsonify(error='Invalid credentials'), 401
    except Exception as e:
        return jsonify(error=str(e)), 500

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
@app.route('/api/detect-anomalies', methods=['POST'])
def api_detect_anomalies():
    data = request.json
    financial_data = np.array(data['financial_data'])  # Assuming 'financial_data' is passed in the request

    anomaly_scores, predictions = detect_anomalies(financial_data)

    return jsonify({
        'anomaly_scores': anomaly_scores.tolist(),
        'predictions': predictions.tolist()
    })

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
data = pd.read_csv('/Users/akshay/Desktop/AIF/dataset.csv')  # Update with the actual path to your dataset

X = data.drop('expense_category', axis=1)
y = data['expense_category']

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

    # Check if debts list is not empty
    if not debts:
        return 0  # No debts to pay off

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
            # Check if indices are within the valid range
            if i - 1 >= 0 and j - 1 >= 0:
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




def evaluate_stl_model(train_data, test_data):
    # Extract the values (assuming train_data and test_data are pandas DataFrames or Series)
    train_data_values = train_data.values.squeeze()
    test_data_values = test_data.values.squeeze()

    # Fit STL decomposition on training data
    stl = STL(train_data_values, seasonal=13)  # You may adjust the seasonal parameter based on your data
    result = stl.fit()

    # Forecast using the trend and seasonal components
    trend, seasonal, _ = result.trend, result.seasonal, result.resid
    predictions = trend + seasonal

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_data_values, predictions))

    return rmse, predictions

# Example usage:
# Assuming 'train' and 'test' are pandas DataFrames or Series
#rmse, predictions = evaluate_stl_model(X_train, y_test)

# Print RMSE
#print(f'Root Mean Squared Error (RMSE) for STL: {rmse}')


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
customer_data = pd.read_csv('your_customer_data.csv')

def k_means_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

# Example usage:
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
