import numpy as np
import pandas as pd
from scipy.stats import norm, stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm

# Record the start time
start_time = time.time()

np.random.seed(12)

def gbm_prices(S, T, d, sigma, mu):
    # Generate random numbers inside the loop
    random_numbers = np.random.normal(0, 1, size=(T*d,))
    prices = [S]
    for i in range(1, T*d):
        price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * (1/d) + sigma * np.sqrt(1/d) * random_numbers[i])
        prices.append(price)
    return prices

def modif_log_likelihood(params):
    mean, std_dev = params
    std_dev = abs(std_dev)

    # Using vectorized operations for conditions
    in_range = (data > L) & (data < U)
    at_L = data == L
    at_U = data == U

    # Log-likelihood for each case
    log_likelihood_in_range = np.log(norm.pdf(data[in_range], loc=mean, scale=std_dev))
    log_likelihood_at_L = np.log(norm.cdf(L, loc=mean, scale=std_dev))
    log_likelihood_at_U = np.log(1 - norm.cdf(U, loc=mean, scale=std_dev))

    # Summing up log-likelihoods
    total_log_likelihood = np.sum(log_likelihood_in_range)
    total_log_likelihood += np.sum(at_L) * log_likelihood_at_L  # Multiply by count of data points at L
    total_log_likelihood += np.sum(at_U) * log_likelihood_at_U  # Multiply by count of data points at U

    # Return negative log-likelihood as we minimize in the optimization
    return -total_log_likelihood

def log_likelihood(params):
    mean, std_dev = params
    std_dev = abs(std_dev)

    # Log-likelihood for each case
    log_likelihood_in_range = np.log(norm.pdf(data, loc=mean, scale=std_dev))

    # Summing up log-likelihoods
    total_log_likelihood = np.sum(log_likelihood_in_range)

    # Return negative log-likelihood as we minimize in the optimization
    return -total_log_likelihood

# Set parameters
S0 = 100  # Initial stock price
T = 1   # Time to expiration in years
d = 250 # Number of trading days per year
mu = 0
sigma = 0.3
L = -0.1
U = 0.1

results = pd.DataFrame(columns=['mu', 'sigma', 'Mu MoM', 'Sigma Mom', 'Mu MLE', 'Sigma MLE'])

# Set parameters
S0 = 100  # Initial stock price
T = 1000    # Time to expiration in years
d = 250 # Number of trading days per year

# Generate stock prices
prices = gbm_prices(S0, T, d, sigma, mu)

# Generate returns from prices
prices_df = pd.DataFrame({'prices': prices})
returns = prices_df['prices'].pct_change().dropna()

# Assuming 'returns' is the Series containing the returns
returns_mean = returns.mean()
returns_std = returns.std()

# Normalize the returns while preserving original mean and standard deviation
data_series = (returns - returns_mean) / returns_std
data_series = data_series * sigma + mu

print(data_series.mean())
print(data_series.std())

data = data_series.values  # Convert Pandas Series to NumPy array

print(data)
data = np.where(data < L, L, data)
data = np.where(data > U, U, data)

initial_guess = [np.mean(data), np.std(data)]
result = minimize(log_likelihood, initial_guess, method='Nelder-Mead')
estimated_mu, estimated_sigma = result.x
estimated_sigma = abs(estimated_sigma)
print("Method of moments ", initial_guess)
print(f"Estimated mean MLE: {estimated_mu}")
print(f"Estimated standard deviation MLE: {estimated_sigma}")
print(f"True mean: {mu}")
print(f"True standard deviation: {sigma}")


