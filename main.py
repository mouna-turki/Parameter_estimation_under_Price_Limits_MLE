import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import time

def gbm_prices(S0, T, d, sigma, mu):
    """Generates stock prices following Geometric Brownian Motion."""
    dt = 1/d
    n_steps = int(T * d)
    random_shocks = np.random.normal(0, 1, n_steps)
    
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    
    for i in range(1, n_steps + 1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[i-1]
        prices[i] = prices[i-1] * np.exp(drift + diffusion)
    return prices

def standard_log_likelihood(params, data):
    """Standard MLE (ignores the fact that data is censored)."""
    mean, std_dev = params
    std_dev = abs(std_dev)
    ll = np.sum(norm.logpdf(data, loc=mean, scale=std_dev))
    return -ll

def censored_log_likelihood(params, data, L, U):
    """
    Censored MLE (Corrects for bias at boundaries L and U).
    This treats L and U as limits rather than exact values.
    """
    mean, std_dev = params
    std_dev = max(abs(std_dev), 1e-6)

    at_L = (data <= L)
    at_U = (data >= U)
    in_range = ~(at_L | at_U)

    # Likelihood for exact values
    ll_in_range = np.sum(norm.logpdf(data[in_range], loc=mean, scale=std_dev))
    
    # Likelihood for values capped at L: P(X <= L)
    ll_at_L = np.sum(np.log(norm.cdf(L, loc=mean, scale=std_dev) + 1e-10))
    
    # Likelihood for values capped at U: P(X >= U)
    ll_at_U = np.sum(np.log(1 - norm.cdf(U, loc=mean, scale=std_dev) + 1e-10))

    return -(ll_in_range + ll_at_L + ll_at_U)

def run_analysis():
    # --- 1. Configuration ---
    np.random.seed(12)
    S0, T, d = 100, 1000, 250
    true_mu, true_sigma = 0, 0.3
    L, U = -0.1, 0.1 
    
    start_time = time.time()

    # --- 2. Data Generation ---
    prices = gbm_prices(S0, T, d, true_sigma, true_mu)
    returns = pd.Series(prices).pct_change().dropna()

    # Normalize returns for testing and apply censoring
    data = (returns - returns.mean()) / returns.std() * true_sigma + true_mu
    data_censored = np.clip(data, L, U)

    # --- 3. Parameter Estimation ---
    initial_guess = [np.mean(data_censored), np.std(data_censored)]

    # A. Standard MLE (Biased)
    res_std = minimize(standard_log_likelihood, initial_guess, args=(data_censored,), method='Nelder-Mead')
    
    # B. Censored MLE (Corrected)
    res_cens = minimize(censored_log_likelihood, initial_guess, args=(data_censored, L, U), method='Nelder-Mead')

    # --- 4. Comparison Results ---
    print(f"--- Quant Finance Analysis ({T} Years) ---")
    print(f"{'Method':<20} | {'Mean':<10} | {'Sigma':<10}")
    print("-" * 45)
    print(f"{'True Values':<20} | {true_mu:<10.4f} | {true_sigma:<10.4f}")
    print(f"{'Method of Moments':<20} | {initial_guess[0]:<10.4f} | {initial_guess[1]:<10.4f}")
    print(f"{'Standard MLE':<20} | {res_std.x[0]:<10.4f} | {abs(res_std.x[1]):<10.4f}")
    print(f"{'Censored MLE':<20} | {res_cens.x[0]:<10.4f} | {abs(res_cens.x[1]):<10.4f}")
    print("-" * 45)
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_analysis()