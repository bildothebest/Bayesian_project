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

cases=cases*90

cases=cases[0:96]
dates=dates[0:96]
indices=[0,34,96]#,204,253,281]
segment_cases=[]
segment_dates=[]

for i,k in zip(indices[:-1], indices[1:]):
    segment_cases.append(cases[i:k])
    segment_dates.append(dates[i:k])
print(segment_cases)
print(segment_dates)


colors = plt.cm.viridis(np.linspace(0, 1, len(segment_dates)))

plt.figure(figsize=(12, 6))

for i, (seg_dates, seg_cases) in enumerate(zip(segment_dates, segment_cases)):
    plt.plot(seg_dates, seg_cases, label=f'Segment {i+1}', color=colors[i], marker='o')

# Aggiungi etichette e legenda
plt.xlabel("Date")
plt.ylabel("Casi cumulativi")
plt.title("Segmentazione dei dati")
plt.legend()
plt.grid()
plt.show()










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
    beta1, beta2, t_change = theta
    # Split time into pre- and post-change point
    t1 = t[t <= t_change]
    t2 = t[t > t_change]
    R0=0
    # Run the SIR model for both segments
    I0 = cases[0]  # Initial number of infected individuals
    try:
        S1, I1, R1 = run_sir_model(t1, beta1, gamma, N, I0, R0)
        S2, I2, R2 = run_sir_model(t2, beta2, gamma, N, I1[-1], R1[-1])
    except:
        return -np.inf

    I_model = np.concatenate([I1, I2])
    sigma = 1.0  # Assumed standard deviation of measurement errors
    # Calculate the log-likelihood
    ll = -0.5 * np.sum(((cases - I_model) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll #log(likelyhood) funzione dei parametri 


def log_prior(theta, t_max):
    beta1, beta2, t_change = theta
    # Uniform prior for beta
    if not (0 < beta1 < 10):
        return -np.inf
    if not (0 < beta2 < 10):
        return -np.inf
    # Uniform prior for gamma
    if not (0 < t_change < t_max):
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
initial_t_change = 34

gamma = 1/9
 
ndim = 3

nwalkers = 50

nsteps = 5000


# =============================================================================
# for i, j in zip(segment_dates, segment_cases):
#     beta_gamma_R0=analisi_del_migaele_e_GPT(i, j, initial_params, population, gamma, ndim, nwalkers, nsteps)
# 
# =============================================================================



#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(dates))
# Initial positions of walkers
initial_pos = np.random.normal(loc=[initial_beta1, initial_beta2, initial_t_change], scale=[0.2, 0.2, 2], size=(nwalkers, ndim))


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
samples=np.column_stack((samples, R0_1, R0_2))
# Plot the posterior distributions
fig = corner.corner(samples, labels=["beta1", "beta2", 't_change', 'R0_1', 'R0_2'])
plt.show()

# Compute and print the mean and standard deviation of the parameters
beta1_mcmc, beta2_mcmc, t_change_mcmc, R0_1_mcmc, R0_2_mcmc = np.mean(samples, axis=0)
beta1_std, beta2_std, t_change_std, R0_1_std, R0_2_std= np.std(samples, axis=0)
print(f"Estimated beta1: {beta1_mcmc:.4f} ± {beta1_std:.4f}")
print(f"Estimated beta2: {beta2_mcmc:.4f} ± {beta2_std:.4f}")
print(f"Estimated t_change: {t_change_mcmc:.4f} ± {t_change_std:.4f}")
print(f"Estimated R0_1: {R0_1_mcmc:.4f} ± {R0_2_std:.4f}")
print(f"Estimated R0_2: {R0_2_mcmc:.4f} ± {R0_2_std:.4f}")


samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
n_posterior_samples = 200
posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]

posterior_predictive = []

for beta1, beta2, t_c in posterior_samples:
    t1 = t[t <= t_c]
    t2 = t[t > t_c]
    I0 = cases[0]
    R0=0
    S1, I1, R1 = run_sir_model(t1, beta1, gamma, population, I0, R0)
    S2, I2, R2 = run_sir_model(t2, beta2, gamma, population, I1[-1], R1[-1])
    posterior_predictive.append(np.concatenate([I1, I2]))

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
plt.axvline(np.median(samples[:, 2]), color="orange", linestyle="--", label="Median $t_c$")
plt.xlabel("Time")
plt.ylabel("Infected Individuals")
plt.legend()
plt.title("Posterior Predictive Plot with $t_c$")
plt.show()



# =============================================================================
# 
# =============================================================================
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
    #return beta_mcmc, gamma_mcmc, R0_mcmc, samples

