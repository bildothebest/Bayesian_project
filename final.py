import numpy as np
import matplotlib.pyplot as plt
from _11_24_migaele_funzioni import switzerland_data
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os
from IPython import get_ipython  # For IPython utilities

get_ipython().run_line_magic('matplotlib', 'inline')


dates, cases=switzerland_data()
dates=dates[0:281]
cases=cases[0:281]
cases=cases*90






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





def log_likelihood(theta, t, N, cases):
    lppd=np.zeros(len(cases))
    betas = theta[:5]  # Estrai i beta
    t_changes = theta[5:9]  # Estrai i t_change
    sigma=theta[9:]
    # Suddividi il tempo in base ai t_change
    t_segments = []
    t_start = t[0]
    for t_change in t_changes:
        t_segments.append(t[(t_start <= t) & (t < t_change)])
        t_start = t_change
    t_segments.append(t[t_start <= t])  # Aggiungi l'ultimo segmento

    # Esegui il modello SIR per ciascun segmento
    I0 = cases[0]  # Numero iniziale di infetti
    R0 = 0
    I_model_segments = []
    sigmas_t=np.concatenate((np.full(int(t_changes[0]), sigma[0]), 
                            np.full(int(t_changes[1])-int(t_changes[0]), sigma[1]), 
                            np.full(int(t_changes[2])-int(t_changes[1]), sigma[2]), 
                            np.full(int(t_changes[3])-int(t_changes[2]), sigma[3]), 
                            np.full(len(cases)-int(t_changes[3]), sigma[4])))
    try:
        for i, t_segment in enumerate(t_segments):
            beta = betas[i]
            S, I, R = run_sir_model(t_segment, beta, gamma, N, I0, R0)
            I_model_segments.append(I)
            I0 = I[-1]  # Aggiorna lo stato iniziale per il segmento successivo
            R0 = R[-1]
    except:
        return -np.inf, np.full(len(cases), -np.inf)

    # Combina i segmenti per ottenere l'intero modello
    I_model = np.concatenate(I_model_segments)
    lppd=-0.5*(cases - I_model)**2 / sigmas_t**2 + np.log(2 * np.pi * sigmas_t**2)
    # Calcola la log-verosimiglianza
    # Deviazione standard assunta
    ll = np.sum(lppd)
    return ll, lppd






def log_prior(theta, t_max):
    betas = theta[:5]  # Estrai i beta
    times = theta[5:9]  # Estrai i t_change
    sigma=theta[9:]
    # Controlla che beta sia in [0, 10]
    if not all(0 < beta < 10 for beta in betas):
        return -np.inf

    # Controlla che i tempi siano ordinati e entro i limiti [0, t_max]
    if not all(0 < times[i] < times[i + 1] for i in range(len(times) - 1)) or (times[-1] > t_max or times[0] < 0):
        return -np.inf

    if not all(5<sigmai<5000 for sigmai in sigma):
        return -np.inf
    return 0



def log_posterior(theta, t, N, cases):
    lp = log_prior(theta, t.max())
    if not np.isfinite(lp):
        return -np.inf, np.full(len(cases), -np.inf)
    ll, lppd = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf, np.full(len(cases), -np.inf)
    return lp + ll, lppd





population = 100000*90

n_betas = 5  # Sostituisci con la lunghezza desiderata
#initial_betas = np.full(n_betas, 1/14)
initial_betas=[1.5/14, 1/28, 1/14, 4/14, 1/56]


def create_intervals(n, array_length):
    # Calcola gli indici che dividono l'array in n-1 intervalli
    indices = np.linspace(0, array_length - 1, n, dtype=int)
    return indices


indices=create_intervals(n_betas+1, len(dates))
print(indices)

initial_t_changes = [34, 80, 204, 253]


gamma = 1/14

initial_sigma=[200, 200, 50, 700, 700]

ndim = n_betas+len(initial_t_changes)+len(initial_sigma)

nwalkers = 50

nsteps = 10000


#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(dates))
# Initial positions of walkers

initial_pos = np.random.normal(
    loc=np.array(list(initial_betas) + list(initial_t_changes) + list(initial_sigma)),
    scale=np.array([0.1] * len(initial_betas) + [2] * len(initial_t_changes) + len(initial_sigma)*[0.5]),
    size=(nwalkers, ndim)
)


# Ensure all initial positions are positive
for i in range(nwalkers):
    for j in range(ndim):
        while initial_pos[i, j] <= 0:
            initial_pos[i, j] = np.random.normal(
                loc=np.array(list(initial_betas) + list(initial_t_changes) +  list(initial_sigma))[j],
                scale=np.array(
                    [0.2] * len(initial_betas) + [2] * len(initial_t_changes) +  len(initial_sigma)*[0.5]
                )[j]
            )

filename = "5beta.h5"
backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, population, cases), backend=backend)


# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(None, nsteps, progress=True)
print("Done.")
#%%
# Get the samples
discard=int(nsteps*0.2)
thin=1

samples = sampler.get_chain(discard=discard, thin=thin, flat=False)
log_prob=sampler.get_log_prob(discard=discard, thin=thin, flat=False)



#Corner no R0
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
labels = [f"beta{i+1}" for i in range(n_betas)] + [f"t_change_{i+1}" for i in range(n_betas-1)]+[f"sigma_{i+1}" for i in range(n_betas)]
fig = corner.corner(samples, labels=labels)
plt.show()



#Model Predictiv
n_posterior_samples = 200
posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]
posterior_samples = samples[-200:]

posterior_predictive = []

for sample in posterior_samples:
    betas=sample[:n_betas]
    times=sample[n_betas:9]

    R0=0
    I0=cases[0]
    times=np.concatenate(([t[0]], times, [t[-1]]))

    single_posterior_predictive=[]

    
    for i in range(n_betas):

        if i == n_betas - 1:  # Ultima iterazione
            t_segment = t[(times[i] <= t) & (t <= times[i + 1])]  # Usa <= per includere l'ultimo punto
        else:
            t_segment = t[(times[i] <= t) & (t < times[i + 1])]  # Condizione originale per tutte le altre iterazioni

        S, I, R=run_sir_model(t_segment, betas[i], gamma, population, I0, R0)
        
        I0=I[-1]
        R0=R[-1]

        single_posterior_predictive.append(I)
        
        
    
    single_posterior_predictive = np.concatenate(single_posterior_predictive)
    posterior_predictive.append(single_posterior_predictive)

posterior_predictive = np.array(posterior_predictive)

mean_prediction = np.mean(posterior_predictive, axis=0)
lower_bound = np.percentile(posterior_predictive, 2.5, axis=0)
upper_bound = np.percentile(posterior_predictive, 97.5, axis=0)

##Plot confidence interval
plt.figure(figsize=(10, 6))
plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)

plt.plot(t, mean_prediction, label="Mean Prediction", color="blue")
plt.fill_between(t, lower_bound, upper_bound, color="blue", alpha=0.3, label="95% CI")
plt.xlabel("Time")
plt.ylabel("Infected Individuals")
plt.legend()
plt.title("Posterior Predictive Plot with $t_c$")
plt.show()

##plot single simulations
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
fig, axes = plt.subplots(15, figsize=(15, 25), sharex=True)
labels = ["beta1","beta2","beta3","beta4","beta5", 't_change1','t_change2','t_change3','t_change4',"sigma1","sigma2","sigma3","sigma4","sigma5"]
for i in range(14):
    ax = axes[i]
    ax.plot(samples_trace[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, samples_trace.shape[0])
    ax.set_ylabel(labels[i])
    ax.grid()

# Plot the log-likelihood trace
log_prob_samples = sampler.get_log_prob(discard=discard, flat=False)
axes[14].plot(log_prob_samples, "k", alpha=0.3)
axes[14].set_ylabel("Log-likelihood")
axes[14].set_xlabel("Step number")
axes[14].grid()

plt.tight_layout()
plt.show()
#return beta_mcmc, gamma_mcmc, R0_mcmc, samples


# =============================================================================
# #WAIC
# =============================================================================
blobs=sampler.get_blobs(discard=discard, thin=thin, flat=False)
timewise_log_likelihood=blobs
lppd_chain = np.log(np.mean(np.exp(timewise_log_likelihood), axis=0))
p_waic_chain = np.var(timewise_log_likelihood, axis=0)
# Check if the any of the terms in p_waic are too large, which indicates
# a problem
if np.any(p_waic_chain > 0.4):
    print(f"Warning: Var[log p(y_i|theta)] > 0.4 for data points "
          f"{np.argwhere(p_waic_chain > 0.4)}. p_WAIC unreliable!")
# Sum up the partitions
lppd = np.sum(lppd_chain)
p_waic = np.sum(p_waic_chain)

waic=-2*(lppd - p_waic)



sampler.acceptance_fraction
#%%  Calculate R0 and do corner 
samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

print(f"Number of samples: {samples.shape[0]}")

R0_list = [samples[:, i] / gamma for i in range(n_betas)]
samples = np.column_stack((samples, *R0_list))

# Etichette per il grafico corner
labels = [f"beta{i+1}" for i in range(n_betas)] + [f't_change_{i+1}' for i in range(n_betas-1)] + [f'R0_{i+1}' for i in range(n_betas)]

# Crea il grafico corner
fig = corner.corner(samples, labels=labels)
plt.show()


# Calcola le medie e deviazioni standard
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)

# Stampa i risultati per i beta
for i in range(n_betas):
    print(f"Estimated beta{i+1}: {means[i]:.4f} ± {stds[i]:.4f}")

# Stampa i risultati per t_change
for i in range(n_betas-1):
    idx = n_betas + i
    print(f"Estimated t_change_{i+1}: {means[idx]:.4f} ± {stds[idx]:.4f}")

# Stampa i risultati per R0
for i in range(n_betas):
    idx = n_betas + (n_betas-1) + i
    print(f"Estimated R0_{i+1}: {means[idx]:.4f} ± {stds[idx]:.4f}")



