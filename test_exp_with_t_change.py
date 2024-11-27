# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:27:10 2024

@author: Ninja
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import arviz as az



# Generate synthetic data
np.random.seed(42)
t = np.linspace(0, 10, 100)  # Time points
true_r1 = 1.1  # Growth rate before tc
true_r2 = -0.3  # Growth rate after tc
true_tc = 5.0  # Change point

# Generate the data
data = np.piecewise(
    t,
    [t <= true_tc, t > true_tc],
    [
        lambda t: np.exp(true_r1 * t),
        lambda t: np.exp(true_r1 * true_tc) * np.exp(true_r2 * (t - true_tc)),
    ],
)

# Add noise
noise = np.random.normal(0, 0.1, size=data.shape)
data_noisy = data + noise

# Plot the data
plt.figure(figsize=(8, 4))
plt.scatter(t, data_noisy, label="Noisy Data", alpha=0.7, s=10)
plt.plot(t, data, label="True Data", color="red", linewidth=2)
plt.axvline(true_tc, color="gray", linestyle="--", label="True Change Point")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# Log-likelihood function
def log_likelihood(params, t, y):
    r1, r2, tc, sigma = params
    if not (0 <= tc <= t.max()):  # Ensure tc is within valid range
        return -np.inf

    # Predicted data
    model = np.piecewise(
        t,
        [t <= tc, t > tc],
        [
            lambda t: np.exp(r1 * t),
            lambda t: np.exp(r1 * tc) * np.exp(r2 * (t - tc)),
        ],
    )
    # Gaussian log-likelihood
    return -0.5 * np.sum(((y - model) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

# Prior function
def log_prior(params):
    r1, r2, tc, sigma = params
    if -5 < r1 < 5 and -5 < r2 < 5 and 0 < sigma < 10 and 0 <= tc <= t.max():
        return 0.0  # Uniform prior
    return -np.inf  # Invalid parameters

# Posterior probability
def log_posterior(params, t, y):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, t, y)

# Initial guess and parameter setup
n_walkers = 32
n_dim = 4  # r1, r2, tc, sigma
initial_guesses = np.random.rand(n_walkers, n_dim) * [1, 1, 10, 0.5] + [0, 0, 0, 0.1]

# Set up the MCMC sampler
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, log_posterior, args=(t, data_noisy)
)

# Run the MCMC
print("Running MCMC...")
sampler.run_mcmc(initial_guesses, 10000, progress=True)

# Extract the samples
samples = sampler.get_chain(discard=2000, thin=10, flat=True)

# Plot results
import corner

fig = corner.corner(
    samples,
    labels=["$r_1$", "$r_2$", "$t_c$", "$\sigma$"],
    truths=[true_r1, true_r2, true_tc, 0.1],
)
plt.show()


import seaborn as sns

# Generate posterior predictive samples
n_posterior_samples = 200  # Number of posterior samples to use
posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]

# Generate model predictions
posterior_predictive = []

for sample in posterior_samples:
    r1, r2, tc, sigma = sample
    model = np.piecewise(
        t,
        [t <= tc, t > tc],
        [
            lambda t: np.exp(r1 * t),
            lambda t: np.exp(r1 * tc) * np.exp(r2 * (t - tc)),
        ],
    )
    posterior_predictive.append(model + np.random.normal(0, sigma, size=t.shape))

posterior_predictive = np.array(posterior_predictive)

# Calculate summary statistics for posterior predictions
mean_prediction = np.mean(posterior_predictive, axis=0)
lower_bound = np.percentile(posterior_predictive, 2.5, axis=0)
upper_bound = np.percentile(posterior_predictive, 97.5, axis=0)

# Extract posterior samples for tc
tc_samples = samples[:, 2]

# Plot the results
plt.figure(figsize=(10, 6))

# Posterior predictive plot
plt.scatter(t, data_noisy, label="Observed Data", color="black", alpha=0.7, s=10)
plt.plot(t, mean_prediction, label="Mean Prediction", color="blue")
plt.fill_between(
    t, lower_bound, upper_bound, color="blue", alpha=0.3, label="95% CI"
)
plt.axvline(true_tc, color="gray", linestyle="--", label="True Change Point")

# Add posterior distribution of tc as a vertical histogram
sns.histplot(tc_samples, color="orange", kde=True, stat="density", bins=30, label="$t_c$ Posterior", alpha=0.4)
plt.axvline(np.median(tc_samples), color="orange", linestyle="-", label="$t_c$ Median")

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("Posterior Predictive Plot with $t_c$ Posterior")
plt.show()


