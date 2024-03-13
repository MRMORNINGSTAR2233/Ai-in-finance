import numpy as np
from scipy.optimize import minimize

def calculate_portfolio_statistics(weights, returns, cov_matrix):
    portfolio_return = np.sum(weights * returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def objective_function(weights, returns, cov_matrix, target_return):
    portfolio_return, portfolio_volatility = calculate_portfolio_statistics(weights, returns, cov_matrix)
    # Minimize the negative of the Sharpe ratio to maximize the Sharpe ratio
    return -(portfolio_return - target_return) / portfolio_volatility

def optimize_portfolio(returns, cov_matrix, target_return, risk_tolerance):
    num_assets = len(returns)
    initial_weights = np.ones(num_assets) / num_assets  # Equal-weighted portfolio as initial guess

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Constraint: weights sum to 1
                   {'type': 'ineq', 'fun': lambda weights: risk_tolerance - calculate_portfolio_statistics(weights, returns, cov_matrix)[1]})  # Constraint: maximum allowed portfolio volatility

    result = minimize(objective_function, initial_weights, args=(returns, cov_matrix, target_return),
                      method='SLSQP', constraints=constraints)

    if result.success:
        optimized_weights = result.x
        return optimized_weights
    else:
        raise ValueError("Portfolio optimization failed.")

# Example usage:
# Assume you have a dataset or user inputs for historical returns and a covariance matrix for a set of assets
returns = np.array([float(x) for x in input("Enter historical returns (comma-separated): ").split(',')])
cov_matrix = np.array([[float(x) for x in input("Enter covariance matrix row (comma-separated): ").split(',')] for _ in range(len(returns))]])

target_return = float(input("Enter your target return: "))
risk_tolerance = float(input("Enter your risk tolerance: "))

optimized_weights = optimize_portfolio(returns, cov_matrix, target_return, risk_tolerance)
print("Optimized Portfolio Weights:", optimized_weights)
