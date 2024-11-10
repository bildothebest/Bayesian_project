# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 04:39:26 2024

@author: Ninja
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import emcee
import matplotlib.pyplot as plt
import os

# Set working directory to the folder where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load observed data (replace 'switzerland_data.csv' with your actual data file)
# The data should contain a column 'Infected' for daily infected cases
data = pd.read_csv("switzerland_data.csv")
observed_cases = data['Infected'].values
time_points = np.arange(len(observed_cases))

# Define the SIR model
def sir_model(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R  # Total population (assumed constant)
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Function to simulate the SIR model over time
def simulate_sir(beta, gamma, initial_conditions, t):
    solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma))
    S, I, R = solution.T
    return I  # Return only the infected population

# Define the log likelihood function
def log_likelihood(params):
    beta, gamma = params
    initial_conditions = (0.99, 0.01, 0)  # Initial fractions of S, I, R, assuming 1% initial infected
    model_predictions = simulate_sir(beta, gamma, initial_conditions, time_points)
    sigma = 10  # Standard deviation of measurement noise (tunable)
    return -0.5 * np.sum(((observed_cases - model_predictions) / sigma) ** 2)

# Define the log prior function
def log_prior(params):
    beta, gamma = params
    if 0 < beta < 1 and 0 < gamma < 1:
        return 0.0  # Uniform prior
    return -np.inf  # Impossible values outside range

# Define the log posterior function
def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# MCMC with emcee
n_walkers = 32
n_iter = 500
initial_beta = 0.2
initial_gamma = 0.1
initial_positions = [np.array([initial_beta, initial_gamma]) + 1e-4 * np.random.randn(2) for _ in range(n_walkers)]

sampler = emcee.EnsembleSampler(n_walkers, 2, log_posterior)
sampler.run_mcmc(initial_positions, n_iter, progress=True)

# Analyze the samples
samples = sampler.get_chain(discard=100, thin=15, flat=True)
beta_samples, gamma_samples = samples[:, 0], samples[:, 1]

# Plot the results
plt.figure(figsize=(10, 5))
plt.hist(beta_samples, bins=30, alpha=0.5, label="beta")
plt.hist(gamma_samples, bins=30, alpha=0.5, label="gamma")
plt.legend()
plt.xlabel("Parameter values")
plt.ylabel("Frequency")
plt.title("Posterior distributions of SIR parameters")
plt.show()
