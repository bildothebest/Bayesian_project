import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os
from funzioni import sir_model, run_sir_model, log_likelihood, log_prior, log_posterior



# Set working directory to the folder where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the data with dates parsed in day-first format
data = pd.read_csv('Asia_Countries_Cases.csv', dayfirst=True)

# Filter data for Switzerland
CH_data = data[data['geoId'] == 'CH']
CH_data['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'] = (
    CH_data['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].astype(float)
)

# Convert 'dateRep' column to datetime if needed and handle potential errors
CH_data.loc[:, 'dateRep'] = pd.to_datetime(CH_data['dateRep'], dayfirst=True, errors='coerce')

# Extract dates and cases
dates = CH_data['dateRep'].values
cases = CH_data['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values

# Invert and slice
dates = dates[::-1][61:]
cases = cases[::-1][61:]

# Plot the observed data
plt.figure(figsize=(12, 6))
plt.plot(dates, cases, 'o', label='Observed cases')
plt.xlabel('Date')
plt.ylabel('Cumulative number of cases')
plt.title('COVID-19 Data in Switzerland')
plt.legend()
plt.grid()
plt.show()

# Find local maxima and minima
window_size = 5

local_maxima_indices = [
    i for i in range(window_size, len(cases) - window_size)
    if cases[i] == max(cases[i - window_size:i + window_size + 1])
]

local_minima_indices = [
    i for i in range(window_size, len(cases) - window_size)
    if cases[i] == min(cases[i - window_size:i + window_size + 1])
]

local_maxima = [(dates[i], cases[i]) for i in local_maxima_indices]
local_minima = [(dates[i], cases[i]) for i in local_minima_indices]



local_maxima=np.array(local_maxima)
local_minima=np.array(local_minima)

all_extrema = np.concatenate((local_maxima, local_minima))

print("estremi")
print(all_extrema)


all_extrema = np.array([
    (pd.Timestamp('2020-04-04'), np.float64(168.77470222)),
    (pd.Timestamp('2020-06-05'), np.float64(2.56304416)),
    (pd.Timestamp('2020-09-21'), np.float64(57.08917533)),
    (pd.Timestamp('2020-11-09'), np.float64(1262.05932757)),
    (pd.Timestamp('2020-12-07'), np.float64(627.51279269))
])



# Estrai solo le date da all_extrema
break_dates = [date for date, _ in all_extrema]



# Trova gli indici corrispondenti a queste date in 'dates'
break_indices = [np.where(dates == date)[0][0] for date in break_dates]

# Usa np.split per dividere dates e cases in segmenti
segments_dates = np.split(dates, break_indices)
segments_cases = np.split(cases, break_indices)

# Stampa i segmenti


for i, (seg_dates, seg_cases) in enumerate(zip(segments_dates, segments_cases)):
    print(f"Segmento {i+1}:")
    print(f"Lunghezza del segmento: {len(seg_dates)}")
    #print("Date:", seg_dates)
    #print("Casi:", seg_cases)
    print()



colors = plt.cm.viridis(np.linspace(0, 1, len(segments_dates)))

plt.figure(figsize=(12, 6))

# Plotta ogni segmento
for i, (seg_dates, seg_cases) in enumerate(zip(segments_dates, segments_cases)):
    plt.plot(seg_dates, seg_cases, label=f'Segment {i+1}', color=colors[i], marker='o')

# Aggiungi etichette e legenda
plt.xlabel("Date")
plt.ylabel("Casi cumulativi")
plt.title("Segmentazione dei dati")
plt.legend()
plt.grid()
plt.show()

###########################################################################################################################
#divisione in segmenti sopra, piccoli fit sotto###########################
###########################################################################################################################

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
    N = len(seg_cases)  # Numero di dati nel segmento
    
    # Creazione del sampler per il segmento
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, N, seg_cases))
    
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


