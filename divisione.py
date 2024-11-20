import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os
from funzioni import sir_model, run_sir_model, log_likelihood, log_prior, log_posterior, switzerland_data, analisi_del_migaele_e_GPT





dates, cases=switzerland_data()

cases=cases*90

 # Plot the observed data
plt.figure(figsize=(12, 6))
plt.plot(dates, cases, 'o', label='Observed cases')
plt.xlabel('Date')
plt.ylabel('Cumulative number of cases')
plt.title('COVID-19 Data in Switzerland')
plt.legend()
plt.grid()
plt.show()





indices=[0,34,96,204,253,281]
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

###########################################################################################################################
#####################################divisione in segmenti sopra, piccoli fit sotto########################################
###########################################################################################################################
population = 100000*90
  
initial_beta = 0.4
initial_gamma = 0.07142
 
ndim = 2

nwalkers = 50

nsteps = 1000

beta_gamma_R0=[]

for i, j in zip(segment_dates, segment_cases):
    beta_gamma_R0=analisi_del_migaele_e_GPT(i, j, initial_beta, population, initial_gamma, ndim, nwalkers, nsteps)










"""
t = np.arange(len(segment_dates[0]))

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
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, segment_cases[0]))

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

S_pred, I_pred, R_pred = run_sir_model(t, best_beta, best_gamma, N, segment_cases[0][0])
C_pred = I_pred  # Cumulative cases predicted

# Plot the observed data and the model prediction
plt.figure(figsize=(12, 6))
plt.plot(segment_dates[0], segment_cases[0], 'o', label='Observed cases')
plt.plot(segment_dates[0], C_pred, '-', label='Predicted cases')
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

"""




"""
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
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, cases_march))



samplers = []  # Per memorizzare i sampler di ciascun segmento

# Itera sui segmenti di cases e date
for i, (seg_dates, seg_cases) in enumerate(zip(segments_dates, segments_cases)):
    print(f"Elaborando segmento {i+1} con {len(seg_cases)} dati...")
    
    # Parametri personalizzati per il segmento
    t = np.arange(len(seg_cases))  # Assumi che t sia un array di tempi per ogni segmento
    n_steps = len(seg_cases)  # Numero di dati nel segmento
    
    # Creazione del sampler per il segmento
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, n_steps, seg_cases))
    
    # Salva il sampler per uso successivo
    samplers.append(sampler)




# Parametri MCMC
nsteps = 1000
discard = int(nsteps * 0.2)

# Lista per salvare i risultati per ogni segmento
results = []

# Itera su ciascun segmento
for i, (seg_dates, seg_cases) in enumerate(zip(segments_dates, segments_cases)):
    print(f"Elaborando segmento {i+1} con {len(seg_cases)} dati...")

    # Parametri del modello per il segmento
    t = np.arange(len(seg_cases))  # Tempo per il segmento
    N = len(seg_cases)  # Popolazione equivalente ai dati
    pos = np.random.rand(nwalkers, ndim)  # Posizione iniziale casuale per gli walker
    
    # Esegui MCMC per il segmento
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, seg_cases))
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    print("Done.")

    # Recupera i campioni
    samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    print(f"Segmento {i+1} - Number of samples: {samples.shape[0]}")
    
    # Calcola R0 e aggiungilo ai campioni
    R0 = samples[:, 0] / samples[:, 1]
    samples = np.column_stack((samples, R0))
    
    # Corner plot dei parametri
    fig = corner.corner(samples, labels=["beta", "gamma", 'R0'], title_fmt=f'Segmento {i+1}')
    plt.show()
    
    # Media e deviazione standard
    beta_mcmc, gamma_mcmc, R0_mcmc = np.mean(samples, axis=0)
    beta_std, gamma_std, R0_std = np.std(samples, axis=0)
    print(f"Segmento {i+1} - Estimated beta: {beta_mcmc:.4f} ± {beta_std:.4f}")
    print(f"Segmento {i+1} - Estimated gamma: {gamma_mcmc:.4f} ± {gamma_std:.4f}")
    print(f"Segmento {i+1} - Estimated R0: {R0_mcmc:.4f} ± {R0_std:.4f}")
    
    # Salva i risultati del segmento
    results.append({
        "segment": i + 1,
        "beta": (beta_mcmc, beta_std),
        "gamma": (gamma_mcmc, gamma_std),
        "R0": (R0_mcmc, R0_std)
    })
    
    # Esegui il modello SIR con i parametri stimati
    best_beta = beta_mcmc
    best_gamma = gamma_mcmc
    S_pred, I_pred, R_pred = run_sir_model(t, best_beta, best_gamma, N, seg_cases[0])
    C_pred = I_pred  # Casi cumulativi predetti

    # Grafico dati osservati e predizione del modello
    plt.figure(figsize=(12, 6))
    plt.plot(seg_dates, seg_cases, 'o', label='Observed cases')
    plt.plot(seg_dates, C_pred, '-', label='Predicted cases')
    plt.xlabel('Date')
    plt.ylabel('Cumulative number of cases')
    plt.title(f'SIR Model Fit - Segmento {i+1}')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Trace plot dei parametri e log-likelihood
    samples_trace = sampler.get_chain(discard=discard, flat=False)
    R0_trace = samples_trace[:, :, 0] / samples_trace[:, :, 1]
    samples_trace_r0 = np.dstack((samples_trace, R0_trace))

    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    labels = ["beta", "gamma", 'R0', "log-likelihood"]
    for j in range(3):
        ax = axes[j]
        ax.plot(samples_trace_r0[:, :, j], "k", alpha=0.3)
        ax.set_xlim(0, samples_trace_r0.shape[0])
        ax.set_ylabel(labels[j])
        ax.grid()

    log_prob_samples = sampler.get_log_prob(discard=discard, flat=False)
    axes[3].plot(log_prob_samples, "k", alpha=0.3)
    axes[3].set_ylabel("Log-likelihood")
    axes[3].set_xlabel("Step number")
    axes[3].grid()

    plt.tight_layout()
    plt.show()

# Risultati finali
for result in results:
    print(f"Segmento {result['segment']}:")
    print(f"  beta: {result['beta'][0]:.4f} ± {result['beta'][1]:.4f}")
    print(f"  gamma: {result['gamma'][0]:.4f} ± {result['gamma'][1]:.4f}")
    print(f"  R0: {result['R0'][0]:.4f} ± {result['R0'][1]:.4f}")
    print()

"""
