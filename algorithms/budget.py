import numpy as np

def initialize_population(population_size, expense_categories, budget_limits):
    population = []
    for _ in range(population_size):
        budget_allocation = {category: np.random.uniform(0, limit) for category, limit in budget_limits.items()}
        population.append(budget_allocation)
    return population

def calculate_fitness(budget_allocation, financial_goals):
    fitness = sum(abs(budget_allocation[category] - financial_goals[category]) for category in budget_allocation)
    return fitness

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

def main():
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

if __name__ == "__main__":
    main()
