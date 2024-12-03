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
    n = (len(theta) + 1) // 2  # Numero di beta
    betas = theta[:n]  # Estrai i beta
    t_changes = theta[n:]  # Estrai i t_change

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
    
    try:
        for i, t_segment in enumerate(t_segments):
            beta = betas[i]
            S, I, R = run_sir_model(t_segment, beta, gamma, N, I0, R0)
            I_model_segments.append(I)
            I0 = I[-1]  # Aggiorna lo stato iniziale per il segmento successivo
            R0 = R[-1]

    except:
        return -np.inf

    # Combina i segmenti per ottenere l'intero modello
    I_model = np.concatenate(I_model_segments)

    # Calcola la log-verosimiglianza
    sigma = 1.0  # Deviazione standard assunta
    ll = -0.5 * np.sum(((cases - I_model) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll






def log_prior(theta, t_max):
    n = (len(theta) + 1) // 2  # Numero di beta
    betas = theta[:n]  # Estrai i beta
    times = theta[n:]  # Estrai i t_change

    # Controlla che beta sia in [0, 10]
    if not all(0 < beta < 10 for beta in betas):
        return -np.inf
    
    # Controlla che i tempi siano ordinati e entro i limiti [0, t_max]
    if not all(0 < times[i] < times[i + 1] for i in range(len(times) - 1)) or (times[-1] > t_max or times[0] < 0):
        return -np.inf
    
    return 0



def log_posterior(theta, t, N, cases):
    lp = log_prior(theta, t.max())
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll





population = 100000*90

n_betas = 5  # Sostituisci con la lunghezza desiderata
initial_betas = np.full(n_betas, 1/14)


def create_intervals(n, array_length):
    # Calcola gli indici che dividono l'array in n-1 intervalli
    indices = np.linspace(0, array_length - 1, n, dtype=int)
    return indices


indices=create_intervals(n_betas+1, len(dates))
print(indices)

initial_t_changes = indices[1:-1]

print(initial_t_changes)

gamma = 1/14
 
ndim = n_betas*2-1

nwalkers = 50

nsteps = 50


# =============================================================================
# for i, j in zip(segment_dates, segment_cases):
#     beta_gamma_R0=analisi_del_migaele_e_GPT(i, j, initial_params, population, gamma, ndim, nwalkers, nsteps)
# 
# =============================================================================



#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(dates))
# Initial positions of walkers
initial_pos = np.random.normal(loc=np.array(list(initial_betas) + list(initial_t_changes)), scale=np.array([0.2]*len(initial_betas) + [2] * len(initial_t_changes)), size=(nwalkers, ndim))


for i in range(nwalkers):
    for j in range(ndim):
        while initial_pos[i, j] <= 0:
            initial_pos[i, j] = np.random.normal(
                loc=np.array(list(initial_betas) + list(initial_t_changes))[j],
                scale=np.array(
                    [0.2] * len(initial_betas) + [2] * len(initial_t_changes)
                )[j]
            )
        




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





samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
labels = [f"beta{i+1}" for i in range(n_betas)] + [f"t_change_{i+1}" for i in range(n_betas-1)]
fig = corner.corner(samples, labels=labels)
plt.show()



n_posterior_samples = 200
posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]





posterior_predictive = []

for sample in posterior_samples:
    betas=sample[:n_betas]
    times=sample[n_betas:]

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

plt.figure(figsize=(10, 6))
plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)

plt.plot(t, mean_prediction, label="Mean Prediction", color="blue")
plt.fill_between(t, lower_bound, upper_bound, color="blue", alpha=0.3, label="95% CI")
plt.xlabel("Time")
plt.ylabel("Infected Individuals")
plt.legend()
plt.title("Posterior Predictive Plot with $t_c$")
plt.show()


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
