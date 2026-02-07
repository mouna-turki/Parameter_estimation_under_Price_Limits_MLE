# GBM Parameter Estimation with Censored Data

This repository contains a quantitative finance tool designed to simulate stock price paths and estimate parameters under market constraints. It specifically focuses on recovering the "true" volatility and drift of a stock when returns are subject to price limits (censoring).

## 📌 Project Overview
In many markets, price limits (limit up/limit down) prevent returns from exceeding certain thresholds ($L$ and $U$). Standard estimation methods like OLS or simple MLE often underestimate volatility in these scenarios because they ignore the data "hidden" beyond the limits.

This project implements:
* **GBM Simulation**: Generates synthetic price paths based on a stochastic differential equation.
* **Censored MLE (Tobit Model)**: A specialized likelihood function that accounts for clipped data at the boundaries.
* **Comparative Analysis**: A head-to-head comparison between Method of Moments, Standard MLE, and Censored MLE.



## 🛠 Features
- **Simulation**: High-performance Geometric Brownian Motion simulation using NumPy.
- **Optimization**: Uses the Nelder-Mead simplex algorithm via SciPy for robust parameter recovery.
- **Stability**: Implements `logpdf` and numerical guards to prevent log-zero errors during optimization.

## 🚀 Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/mouna-turki/Parameter_estimation_under_Price_Limits_MLE.git](https://github.com/mouna-turki/Parameter_estimation_under_Price_Limits_MLE.git)
   cd Parameter_estimation_under_Price_Limits_MLE
2. Install dependencies
    pip install -r requirements.txt
3. Running the analysis
    python main.py
    
