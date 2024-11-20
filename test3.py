

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os
from funzioni import sir_model, run_sir_model, log_likelihood, log_prior, log_posterior, switzerland_data


dates, cases=switzerland_data()




dates_march=dates[0:31]
cases_march=cases[0:31]*90
# Time points in days


t = np.arange(len(cases_march))

# Total population
N = 100000*90



# Initial guesses for beta and gamma
initial_beta = 0.4
initial_gamma = 0.07142

# Number of dimensions (parameters)
ndim = 2

# Number of walkers
nwalkers = 50

# Initial positions of walkers
pos = [np.array([initial_beta, initial_gamma]) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, cases_march))

# Number of steps
nsteps = 1000

discard=int(nsteps*0.2)
# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, progress=True)
print("Done.")

# Get the samples
samples = sampler.get_chain(discard=discard, thin=15, flat=True)
print(f"Number of samples: {samples.shape[0]}")
R0=samples[:,0]/samples[:,1]
samples=np.column_stack((samples, R0))
# Plot the posterior distributions
fig = corner.corner(samples, labels=["beta", "gamma", 'R0'])
plt.show()

# Compute and print the mean and standard deviation of the parameters
beta_mcmc, gamma_mcmc, R0_mcmc = np.mean(samples, axis=0)
beta_std, gamma_std, R0_std = np.std(samples, axis=0)
print(f"Estimated beta: {beta_mcmc:.4f} ± {beta_std:.4f}")
print(f"Estimated gamma: {gamma_mcmc:.4f} ± {gamma_std:.4f}")
print(f"Estimated R0: {R0_mcmc:.4f} ± {R0_std:.4f}")

# Run the SIR model with estimated parameters
best_beta = beta_mcmc
best_gamma = gamma_mcmc

S_pred, I_pred, R_pred = run_sir_model(t, best_beta, best_gamma, N, cases_march[0])
C_pred = I_pred  # Cumulative cases predicted

# Plot the observed data and the model prediction
plt.figure(figsize=(12, 6))
plt.plot(dates_march, cases_march, 'o', label='Observed cases')
plt.plot(dates_march, C_pred, '-', label='Predicted cases')
plt.xlabel('Date')
plt.ylabel('Cumulative number of cases')
plt.title('SIR Model Fit to COVID-19 Data in Switzerland')
plt.legend()
plt.grid()
plt.show()



# Extract the samples for each parameter after burn-in (discard the first 200 steps)
samples_trace = sampler.get_chain(discard=discard, flat=False)
# Calculate the new third column as element-wise division of the first column by the second
R0 = samples_trace[:, :, 0] / samples_trace[:, :, 1]

# Stack the third column to create a new array with shape [800, 50, 3]
samples_trace_r0 = np.dstack((samples_trace, R0))
# Trace plot for each parameter
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
labels = ["beta", "gamma", 'R0',"log-likelihood"]
for i in range(3):
    ax = axes[i]
    ax.plot(samples_trace_r0[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, samples_trace_r0.shape[0])
    ax.set_ylabel(labels[i])
    ax.grid()

# Plot the log-likelihood trace
log_prob_samples = sampler.get_log_prob(discard=discard, flat=False)
axes[3].plot(log_prob_samples, "k", alpha=0.3)
axes[3].set_ylabel("Log-likelihood")
axes[3].set_xlabel("Step number")
axes[3].grid()

plt.tight_layout()
plt.show()





# =============================================================================
# 
# # Extract the samples for each parameter after burn-in (discard the first 200 steps)
# samples_trace = sampler.get_chain(discard=discard, flat=False)
# 
# # Trace plot for each parameter
# fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
# labels = ["beta", "gamma", "log-likelihood"]
# for i in range(2):
#     ax = axes[i]
#     ax.plot(samples_trace[:, :, i], "k", alpha=0.3)
#     ax.set_xlim(0, samples_trace.shape[0])
#     ax.set_ylabel(labels[i])
#     ax.grid()
# 
# # Plot the log-likelihood trace
# log_prob_samples = sampler.get_log_prob(discard=discard, flat=False)
# axes[2].plot(log_prob_samples, "k", alpha=0.3)
# axes[2].set_ylabel("Log-likelihood")
# axes[2].set_xlabel("Step number")
# axes[2].grid()
# 
# # Plot R0 trace
# axes[3].plot(R0, "k", alpha=0.3)
# axes[3].set_ylabel('R0')
# axes[3].set_xlabel("Step number")
# axes[3].grid()
# 
# plt.tight_layout()
# plt.show()
# 
# =============================================================================

