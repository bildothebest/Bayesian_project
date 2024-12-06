# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:38:55 2024

@author: Ninja
"""

import numpy as np
import matplotlib.pyplot as plt
from _11_24_migaele_funzioni import switzerland_data
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os




dates, cases=switzerland_data()
dates=dates[0:281]
cases=cases[0:281]
cases=cases*90

# =============================================================================
# segment_cases=[]
# segment_dates=[]
# 
# for i,k in zip(indices[:-1], indices[1:]):
#     segment_cases.append(cases[i:k])
#     segment_dates.append(dates[i:k])
# print(segment_cases)
# print(segment_dates)
# 
# 
# colors = plt.cm.viridis(np.linspace(0, 1, len(segment_dates)))
# 
# plt.figure(figsize=(12, 6))
# 
# for i, (seg_dates, seg_cases) in enumerate(zip(segment_dates, segment_cases)):
#     plt.plot(seg_dates, seg_cases, label=f'Segment {i+1}', color=colors[i], marker='o')
# 
# # Aggiungi etichette e legenda
# plt.xlabel("Date")
# plt.ylabel("Casi cumulativi")
# plt.title("Segmentazione dei dati")
# plt.legend()
# plt.grid()
# plt.show()
# 
# =============================================================================









# SIR model differential equations
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt] #Sistema di differenziali

# Function to integrate the SIR equations
def run_sir_model(t, beta, gamma, N, I0, R0):
    S0 = N - I0
    y0 = [S0, I0, R0]
    ret = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    return S, I, R #Soluzioni delle differenziali given initial conditions

# Log-likelihood function
def log_likelihood(theta, t, N, cases):
    beta1, beta2, beta3, beta4, beta5, t_change1,t_change2,t_change3,t_change4  = theta
    # Split time into pre- and post-change point
    t1 = t[t <= t_change1]
    t2 = t[(t_change1 < t) & (t < t_change2)]
    t3 = t[(t_change2 < t) & (t < t_change3)]
    t4 = t[(t_change3 < t) & (t < t_change4)]
    t5 = t[t >= t_change4]



    R0=0
    # Run the SIR model for both segments
    I0 = cases[0]  # Initial number of infected individuals
    try:
        S1, I1, R1 = run_sir_model(t1, beta1, gamma, N, I0, R0)
        S2, I2, R2 = run_sir_model(t2, beta2, gamma, N, I1[-1], R1[-1])
        S3, I3, R3 = run_sir_model(t3, beta3, gamma, N, I2[-1], R2[-1])
        S4, I4, R4 = run_sir_model(t4, beta4, gamma, N, I3[-1], R3[-1])
        S5, I5, R5 = run_sir_model(t5, beta5, gamma, N, I4[-1], R4[-1])

    except:
        return -np.inf

    I_model = np.concatenate([I1, I2, I3, I4, I5])
    sigma = 1.0  # Assumed standard deviation of measurement errors
    # Calculate the log-likelihood
    ll = -0.5 * np.sum(((cases - I_model) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll #log(likelyhood) funzione dei parametri 


def log_prior(theta, t_max):
    beta1, beta2, beta3, beta4, beta5, t_change1,t_change2,t_change3,t_change4  = theta
    # Uniform prior for beta
    if not all(0 < beta < 10 for beta in [beta1, beta2, beta3, beta4, beta5]):
        return -np.inf
    # Uniform prior for gamma
    if not (0 < t_change1< t_change2<t_change3<t_change4< t_max):
        return -np.inf
    return 0  # Combine priors for beta and gamma


# Log-posterior fuction
def log_posterior(theta, t, N, cases):
    lp = log_prior(theta, t.max())
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll




###########################################################################################################################
#####################################divisione in segmenti sopra, piccoli fit sotto########################################
###########################################################################################################################
population = 100000*90
  
initial_beta1 = 0.4
initial_beta2 = 0.4
initial_beta3 = 0.4
initial_beta4 = 0.4
initial_beta5 = 0.4

initial_betas=np.array((initial_beta1, initial_beta2, initial_beta3, initial_beta4, initial_beta5))
indices=[0,34,96,204,253,281]
initial_t_changes = np.array((34,96,204,253))

gamma = 1/9
 
ndim = 9

nwalkers = 50

nsteps = 1000


# =============================================================================
# for i, j in zip(segment_dates, segment_cases):
#     beta_gamma_R0=analisi_del_migaele_e_GPT(i, j, initial_params, population, gamma, ndim, nwalkers, nsteps)
# 
# =============================================================================



#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(dates))
# Initial positions of walkers
initial_pos = np.random.normal(loc=np.array(list(initial_betas) + list(initial_t_changes)), scale=np.array([0.2]*len(initial_betas) + [2] * len(initial_t_changes)), size=(nwalkers, ndim))


# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, population, cases))



# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(initial_pos, nsteps, progress=True)
print("Done.")

# Get the samples
discard=int(nsteps*0.2)
thin=1
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
print(f"Number of samples: {samples.shape[0]}")
R0_1=samples[:,0]/gamma
R0_2=samples[:,1]/gamma
R0_3=samples[:,2]/gamma
R0_4=samples[:,3]/gamma
R0_5=samples[:,4]/gamma
samples=np.column_stack((samples, R0_1, R0_2, R0_3, R0_4, R0_5))
# Plot the posterior distributions
fig = corner.corner(samples, labels=["beta1", "beta2","beta3", "beta4", "beta5",'t_change_1','t_change_2','t_change_3','t_change_4', 'R0_1', 'R0_2', 'R0_3', 'R0_4', 'R0_5'])
plt.show()

# Compute and print the mean and standard deviation of the parameters
beta1_mcmc, beta2_mcmc,beta3_mcmc, beta4_mcmc,beta5_mcmc, t_change_1_mcmc, t_change_2_mcmc, t_change_3_mcmc, t_change_4_mcmc, R0_1_mcmc, R0_2_mcmc,R0_3_mcmc, R0_4_mcmc,R0_5_mcmc = np.mean(samples, axis=0)
beta1_std, beta2_std,beta3_std, beta4_std,beta5_std, t_change_1_std, t_change_2_std, t_change_3_std, t_change_4_std, R0_1_std, R0_2_std,R0_3_std, R0_4_std,R0_5_std= np.std(samples, axis=0)
print(f"Estimated beta1: {beta1_mcmc:.4f} ± {beta1_std:.4f}")
print(f"Estimated beta2: {beta2_mcmc:.4f} ± {beta2_std:.4f}")
print(f"Estimated beta3: {beta3_mcmc:.4f} ± {beta3_std:.4f}")
print(f"Estimated beta4: {beta4_mcmc:.4f} ± {beta4_std:.4f}")
print(f"Estimated beta5: {beta5_mcmc:.4f} ± {beta5_std:.4f}")

print(f"Estimated t_change_1: {t_change_1_mcmc:.4f} ± {t_change_1_std:.4f}")
print(f"Estimated t_change_2: {t_change_2_mcmc:.4f} ± {t_change_2_std:.4f}")
print(f"Estimated t_change_3: {t_change_3_mcmc:.4f} ± {t_change_3_std:.4f}")
print(f"Estimated t_change_4: {t_change_4_mcmc:.4f} ± {t_change_4_std:.4f}")

print(f"Estimated R0_1: {R0_1_mcmc:.4f} ± {R0_1_std:.4f}")
print(f"Estimated R0_2: {R0_2_mcmc:.4f} ± {R0_2_std:.4f}")
print(f"Estimated R0_3: {R0_3_mcmc:.4f} ± {R0_3_std:.4f}")
print(f"Estimated R0_4: {R0_4_mcmc:.4f} ± {R0_4_std:.4f}")
print(f"Estimated R0_5: {R0_5_mcmc:.4f} ± {R0_5_std:.4f}")


samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
#fig = corner.corner(samples, labels=["beta1", "beta2","beta3", "beta4", "beta5",'t_change_1','t_change_2','t_change_3','t_change_4'])
#plt.show()
n_posterior_samples = 200
posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]

posterior_predictive = []

for beta1, beta2, beta3, beta4, beta5, t_change1, t_change2, t_change3, t_change4 in posterior_samples:
    t1 = t[t <= t_change1]
    t2 = t[(t_change1 < t) & (t < t_change2)]
    t3 = t[(t_change2 < t) & (t < t_change3)]
    t4 = t[(t_change3 < t) & (t < t_change4)]
    t5 = t[t >= t_change4]
    R0=0
    # Run the SIR model for both segments
    I0 = cases[0]  # Initial number of infected individuals
    
    S1, I1, R1 = run_sir_model(t1, beta1, gamma, population, I0, R0)
    S2, I2, R2 = run_sir_model(t2, beta2, gamma, population, I1[-1], R1[-1])
    S3, I3, R3 = run_sir_model(t3, beta3, gamma, population, I2[-1], R2[-1])
    S4, I4, R4 = run_sir_model(t4, beta4, gamma, population, I3[-1], R3[-1])
    S5, I5, R5 = run_sir_model(t5, beta5, gamma, population, I4[-1], R4[-1])
    
    posterior_predictive.append(np.concatenate([I1, I2, I3, I4, I5]))

posterior_predictive = np.array(posterior_predictive)
# Summary statistics
mean_prediction = np.mean(posterior_predictive, axis=0)
lower_bound = np.percentile(posterior_predictive, 2.5, axis=0)
upper_bound = np.percentile(posterior_predictive, 97.5, axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)
plt.plot(t, mean_prediction, label="Mean Prediction", color="blue")
plt.fill_between(t, lower_bound, upper_bound, color="blue", alpha=0.3, label="95% CI")
plt.xlabel("Time")
plt.ylabel("Infected Individuals")
plt.legend()
plt.title("Posterior Predictive Plot with $t_c$")
plt.show()

# Create the plot
plt.figure(figsize=(12, 6))
for trajectory in posterior_predictive:
    plt.plot(t, trajectory, color="blue", alpha=0.1)  # Individual trajectories

# Highlight the mean trajectory
mean_trajectory = np.mean(posterior_predictive, axis=0)
plt.plot(t, mean_trajectory, color="red", label="Mean Trajectory", linewidth=2)
plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)

# Add labels, legend, and title
plt.xlabel("Time")
plt.ylabel("Infected Individuals")
plt.title("Posterior Predictive Trajectories")
plt.legend()
plt.grid()
plt.show()

# =============================================================================
# 
# =============================================================================
# Extract the samples for each parameter after burn-in (discard the first 200 steps)
samples_trace = sampler.get_chain(discard=discard, flat=False)

# Trace plot for each parameter
fig, axes = plt.subplots(10, figsize=(10, 7), sharex=True)
labels = ["beta1","beta2","beta3","beta4","beta5", 't_change1','t_change2','t_change3','t_change4',"log-likelihood"]
for i in range(9):
    ax = axes[i]
    ax.plot(samples_trace[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, samples_trace.shape[0])
    ax.set_ylabel(labels[i])
    ax.grid()

# Plot the log-likelihood trace
log_prob_samples = sampler.get_log_prob(discard=discard, flat=False)
axes[9].plot(log_prob_samples, "k", alpha=0.3)
axes[9].set_ylabel("Log-likelihood")
axes[9].set_xlabel("Step number")
axes[9].grid()

plt.tight_layout()
plt.show()
    #return beta_mcmc, gamma_mcmc, R0_mcmc, samples

