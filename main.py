import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import time

def gbm_prices(S0, T, d, sigma, mu):
    """
    Simulates stock price paths using Geometric Brownian Motion.
    Formula: S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    """
    dt = 1/d
    n_steps = int(T * d)
    random_shocks = np.random.normal(0, 1, n_steps)
    
    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    
    for i in range(1, n_steps + 1):
        # We use i-1 for random_shocks because it is 0-indexed
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_shocks[i-1]
        prices[i] = prices[i-1] * np.exp(drift + diffusion)
        
    return prices

def log_likelihood(params, data):
    """
    Standard Negative Log-Likelihood for a Normal Distribution.
    Used for MLE estimation of mean and sigma.
    """
    mean, std_dev = params
    std_dev = abs(std_dev) # Ensure volatility stays positive during optimization
    
    # We use the log of the PDF to calculate log-likelihood
    ll = np.sum(norm.logpdf(data, loc=mean, scale=std_dev))
    return -ll

def run_simulation():
    # --- 1. Configuration ---
    np.random.seed(12)
    S0, T, d = 100, 1000, 250  # 1000 years of daily data
    true_mu, true_sigma = 0, 0.3
    L, U = -0.1, 0.1 # Clipping boundaries
    
    start_time = time.time()

    # --- 2. Data Generation ---
    prices = gbm_prices(S0, T, d, true_sigma, true_mu)
    prices_ser = pd.Series(prices)
    
    # Calculate daily returns
    returns = prices_ser.pct_change().dropna()

    # Normalize returns to match our target mu/sigma for testing
    data = (returns - returns.mean()) / returns.std() * true_sigma + true_mu
    
    # Apply Censoring (Clipping)
    # This simulates a market with price limits
    data_censored = np.clip(data, L, U)

    # --- 3. Parameter Estimation ---
    # Method of Moments (Initial Guess)
    mom_mean = np.mean(data_censored)
    mom_std = np.std(data_censored)
    initial_guess = [mom_mean, mom_std]

    # Maximum Likelihood Estimation
    result = minimize(
        log_likelihood, 
        initial_guess, 
        args=(data_censored,), 
        method='Nelder-Mead'
    )
    
    est_mu, est_sigma = result.x

    # --- 4. Output Results ---
    print(f"--- Simulation Results ({T} Years) ---")
    print(f"True Values:      Mean: {true_mu:.4f}, Std: {true_sigma:.4f}")
    print(f"Moments (Clipped): Mean: {mom_mean:.4f}, Std: {mom_std:.4f}")
    print(f"MLE Estimates:    Mean: {est_mu:.4f}, Std: {abs(est_sigma):.4f}")
    print(f"Execution Time:   {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_simulation()