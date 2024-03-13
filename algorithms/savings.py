import numpy as np

class SavingsGoalAgent:
    def __init__(self, goal_amount, initial_savings, monthly_income):
        self.goal_amount = goal_amount
        self.savings = initial_savings
        self.monthly_income = monthly_income
        self.epsilon = 0.2  # Exploration-exploitation trade-off parameter
        self.alpha = 0.1    # Learning rate

    def take_action(self):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.epsilon:
            return np.random.uniform(0, self.monthly_income)
        else:
            return self.get_recommendation()

    def get_recommendation(self):
        # Choose action based on current knowledge
        return min(self.monthly_income, self.goal_amount - self.savings)

    def update_q_value(self, action, reward):
        # Q-learning update rule
        self.savings += action
        self.savings = min(self.savings, self.goal_amount)  # Cap savings to the goal amount
        self.savings = max(self.savings, 0)  # Ensure savings is non-negative
        self.savings_percentage = self.savings / self.goal_amount
        self.q_value = reward + self.alpha * (self.goal_amount - self.savings)

def savings_goal_reinforcement_learning():
    # User inputs
    goal_amount = float(input("Enter your savings goal amount: "))
    initial_savings = float(input("Enter your initial savings: "))
    monthly_income = float(input("Enter your monthly income: "))
    num_episodes = int(input("Enter the number of training episodes: "))

    agent = SavingsGoalAgent(goal_amount, initial_savings, monthly_income)

    for episode in range(num_episodes):
        # Agent takes action
        action = agent.take_action()

        # Calculate reward (negative of the remaining savings towards the goal)
        reward = -(goal_amount - agent.savings)

        # Update Q-value and agent's savings based on the taken action and reward
        agent.update_q_value(action, reward)

    return agent.savings_percentage

# Example usage:
final_savings_percentage = savings_goal_reinforcement_learning()
print(f"Final savings percentage towards the goal: {final_savings_percentage * 100}%")
