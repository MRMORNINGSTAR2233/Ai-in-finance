import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_simulation(initial_investment, annual_return, volatility, years, num_simulations):
    simulation_results = []

    for _ in range(num_simulations):
        annual_returns = np.random.normal(annual_return, volatility, size=years)
        cumulative_returns = np.cumprod(1 + annual_returns)
        simulated_portfolio_value = initial_investment * cumulative_returns[-1]
        simulation_results.append(simulated_portfolio_value)

    return simulation_results

def plot_simulation(simulation_results):
    plt.figure(figsize=(10, 6))
    plt.hist(simulation_results, bins=30, density=True, edgecolor='black')
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Probability Density')
    plt.show()

def main():
    # Parameters
    initial_investment = 100000  # Initial investment amount
    annual_return = 0.08  # Expected annual return
    volatility = 0.15  # Annual volatility (standard deviation of returns)
    years = 10  # Number of years for simulation
    num_simulations = 1000  # Number of simulations

    # Run Monte Carlo Simulation
    simulation_results = monte_carlo_simulation(initial_investment, annual_return, volatility, years, num_simulations)

    # Plot the results
    plot_simulation(simulation_results)

    # Calculate and print statistics
    mean_portfolio_value = np.mean(simulation_results)
    median_portfolio_value = np.median(simulation_results)
    success_rate = np.mean(simulation_results >= initial_investment)
    
    print(f"Mean Portfolio Value: ${mean_portfolio_value:.2f}")
    print(f"Median Portfolio Value: ${median_portfolio_value:.2f}")
    print(f"Probability of Success: {success_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
