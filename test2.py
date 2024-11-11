# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 04:47:01 2024

@author: Ninja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os

# Set working directory to the folder where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load the data
data = pd.read_csv('switzerland_data.csv')
# Ensure the data has columns 'Date' and 'Cases'
dates = pd.to_datetime(data['Date'])
cases = data['Infected'].values
t = np.arange(len(cases))  # Time points in days

# Total population, N (Switzerland's population)
N = 8_600_000

# SIR model differential equations
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to integrate the SIR equations
def run_sir_model(t, beta, gamma, N, I0):
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    ret = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    return S, I, R

# Log-likelihood function
def log_likelihood(theta, t, N, cases):
    beta, gamma = theta
    # Check for valid parameter values
    if beta <= 0 or gamma <= 0 or beta >= 1 or gamma >= 1:
        return -np.inf
    I0 = cases[0]  # Initial number of infected individuals
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    try:
        ret = odeint(sir_model, y0, t, args=(beta, gamma, N))
        S, I, R = ret.T
    except:
        return -np.inf
    C = N - S  # Cumulative cases
    sigma = 1.0  # Assumed standard deviation of measurement errors
    # Calculate the log-likelihood
    ll = -0.5 * np.sum(((cases - C) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll


def log_prior(theta):
    beta, gamma = theta
    # Uniform prior for beta
    if not (0 < beta < 1):
        return -np.inf
    # Gaussian prior for gamma centered at 1/7 with a standard deviation of 0.05
    gamma_mean = 1 / 7
    gamma_std = 0.05
    log_prior_gamma = norm.logpdf(gamma, gamma_mean, gamma_std)
    return log_prior_gamma  # Combine priors for beta and gamma


# Log-posterior function
def log_posterior(theta, t, N, cases):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# Initial guesses for beta and gamma
initial_beta = 0.4
initial_gamma = 0.1

# Number of dimensions (parameters)
ndim = 2

# Number of walkers
nwalkers = 50

# Initial positions of walkers
pos = [np.array([initial_beta, initial_gamma]) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, cases))

# Number of steps
nsteps = 1000

# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, progress=True)
print("Done.")

# Get the samples
samples = sampler.get_chain(discard=200, thin=15, flat=True)
print(f"Number of samples: {samples.shape[0]}")

# Plot the posterior distributions
fig = corner.corner(samples, labels=["beta", "gamma"], truths=[initial_beta, initial_gamma])
plt.show()

# Compute and print the mean and standard deviation of the parameters
beta_mcmc, gamma_mcmc = np.mean(samples, axis=0)
beta_std, gamma_std = np.std(samples, axis=0)
print(f"Estimated beta: {beta_mcmc:.4f} ± {beta_std:.4f}")
print(f"Estimated gamma: {gamma_mcmc:.4f} ± {gamma_std:.4f}")

# Run the SIR model with estimated parameters
best_beta = beta_mcmc
best_gamma = gamma_mcmc

S_pred, I_pred, R_pred = run_sir_model(t, best_beta, best_gamma, N, cases[0])
C_pred = N - S_pred  # Cumulative cases predicted

# Plot the observed data and the model prediction
plt.figure(figsize=(12, 6))
plt.plot(dates, cases, 'o', label='Observed cases')
plt.plot(dates, C_pred, '-', label='Predicted cases')
plt.xlabel('Date')
plt.ylabel('Cumulative number of cases')
plt.title('SIR Model Fit to COVID-19 Data in Switzerland')
plt.legend()
plt.grid()
plt.show()



# Extract the samples for each parameter after burn-in (discard the first 200 steps)
samples = sampler.get_chain(discard=int(nsteps*0.2), flat=False)

# Trace plot for each parameter
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
labels = ["beta", "gamma", "log-likelihood"]
for i in range(2):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, samples.shape[0])
    ax.set_ylabel(labels[i])
    ax.grid()

# Plot the log-likelihood trace
log_prob_samples = sampler.get_log_prob(discard=int(nsteps*0.2), flat=False)
axes[2].plot(log_prob_samples, "k", alpha=0.3)
axes[2].set_ylabel("Log-likelihood")
axes[2].set_xlabel("Step number")
axes[2].grid()

plt.tight_layout()
plt.show()

