import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# Settings
tickers = ['AAPL', 'AMZN', 'GOOG', 'JNJ', 'JPM', 'MSFT', 'TSLA']
start_date = '2022-01-01'
end_date = '2024-01-01'

# Download data
data = yf.download(tickers, start=start_date, end=end_date)
print(f"Data downloaded with shape: {data.shape}")

# Use Close prices (data has MultiIndex columns like ('Close', 'AAPL'))
close_prices = data['Close'].dropna()

# Compute daily returns
returns = close_prices.pct_change().dropna()

# Calculate expected returns (annualized)
mean_returns = returns.mean() * 252
print("\nAnnualized Expected Returns:")
print((mean_returns * 100).round(2).astype(str) + '%')

# Covariance matrix (annualized)
cov_matrix = returns.cov() * 252

# Number of assets
n = len(tickers)

# Define optimization variables
weights = cp.Variable(n)

# Portfolio return and variance expressions
portfolio_return = mean_returns.values @ weights
portfolio_variance = cp.quad_form(weights, cov_matrix.values)

# Constraints
constraints = [
    cp.sum(weights) == 1,       # Full investment
    weights >= 0,               # No shorting
    weights <= 0.3              # Max 30% per asset
]

# Objective: minimize variance - return (maximize return for given risk aversion)
risk_aversion = 0.5
objective = cp.Minimize(risk_aversion * portfolio_variance - (1 - risk_aversion) * portfolio_return)

# Solve problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)

# Check solver status
if problem.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError(f"Solver did not find an optimal solution: status = {problem.status}")

opt_weights = weights.value
if opt_weights is None:
    raise ValueError("Optimization failed: no solution found.")

# Display results
print("\nOptimized Portfolio Weights:")
for ticker, weight in zip(tickers, opt_weights):
    print(f"{ticker}: {weight * 100:.2f}%")

final_return = mean_returns.values @ opt_weights
final_risk = np.sqrt(opt_weights.T @ cov_matrix.values @ opt_weights)

print(f"Expected Annual Return: {final_return * 100:.2f}%")
print(f"Expected Annual Risk (Std Dev): {final_risk * 100:.2f}%")

# Plot Efficient Frontier by varying risk aversion parameter
def efficient_frontier(mean_returns, cov_matrix, n_points=50):
    mus = []
    sigmas = []
    weights_list = []
    for r in np.linspace(0, 1, n_points):
        w = cp.Variable(n)
        ret = mean_returns.values @ w
        var = cp.quad_form(w, cov_matrix.values)
        constr = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.3
        ]
        prob = cp.Problem(cp.Minimize(r * var - (1 - r) * ret), constr)
        prob.solve(solver=cp.SCS)
        if w.value is not None:
            mus.append(mean_returns.values @ w.value)
            sigmas.append(np.sqrt(w.value.T @ cov_matrix.values @ w.value))
            weights_list.append(w.value)
    return np.array(sigmas), np.array(mus), weights_list

sigmas, mus, weights_list = efficient_frontier(mean_returns, cov_matrix)

plt.figure(figsize=(10, 6))
plt.plot(sigmas * 100, mus * 100, 'b-o', label='Efficient Frontier')
plt.xlabel('Annualized Risk (Std Dev %)') 
plt.ylabel('Annualized Return (%)')
plt.title('Efficient Frontier with Constraints (No shorting, max 30% per asset)')
plt.grid(True)

# Highlight optimized portfolio point
plt.scatter(final_risk * 100, final_return * 100, c='red', marker='*', s=200, label='Optimized Portfolio')
plt.legend()
plt.show()
