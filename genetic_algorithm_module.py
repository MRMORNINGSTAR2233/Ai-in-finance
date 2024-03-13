import numpy as np

def genetic_algorithm(expense_categories, budget_limits, financial_goals,
                      population_size=10, num_generations=100, mutation_rate=0.1):
    num_categories = len(expense_categories)
    
    # Generate an initial random population of budget allocations
    initial_population = np.random.uniform(0, 1, size=(population_size, num_categories))

    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = fitness_function(initial_population, budget_limits, financial_goals)

        # Select parents based on their fitness scores (higher fitness, higher chance of being selected)
        selected_parents_indices = np.random.choice(population_size, size=population_size // 2, p=fitness_scores,replace=True)

        # Crossover: Create new individuals by combining the genes of selected parents
        crossover_point = np.random.randint(1, num_categories)
        crossovered_population = crossover(initial_population[selected_parents_indices], crossover_point)

        # Mutation: Introduce small random changes to some individuals
        mutated_population = mutate(crossovered_population, mutation_rate)

        # Replace the old population with the new one
        initial_population = mutated_population

    # Select the best solution from the final generation as the optimized budget allocation
    best_solution_index = np.argmax(fitness_function(initial_population, budget_limits, financial_goals))
    optimized_allocation = initial_population[best_solution_index]

    return optimized_allocation

def fitness_function(population, budget_limits, financial_goals):
    # Calculate fitness as the sum of budget allocations
    fitness = np.sum(population, axis=1)

    # Apply penalty for exceeding budget limits or not meeting financial goals
    print(budget_limits.keys(), financial_goals.keys())
    for i in range(len(population)):
        if any(population[i] > 10000):
            fitness[i] -= 1000  # Penalize if budget limits are exceeded
        if any(population[i] < 10000):
            fitness[i] -= 1000  # Penalize if financial goals are not met

    return fitness

def crossover(parents, crossover_point):
    # Create new individuals by combining genes of parents up to the crossover point
    crossovered_population = np.concatenate((parents[:, :crossover_point], parents[:, crossover_point:]), axis=1)
    return crossovered_population

def mutate(population, mutation_rate):
    # Introduce random changes to the population based on the mutation rate
    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    mutated_population = population + mutation_mask * np.random.uniform(-0.1, 0.1, size=population.shape)
    return mutated_population
