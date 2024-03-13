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
