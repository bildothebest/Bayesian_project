import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os





# SIR model differential equations
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt] #Sistema di differenziali

# Function to integrate the SIR equations
def run_sir_model(t, beta, gamma, N, I0):
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    ret = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    return S, I, R #Soluzioni delle differenziali given initial conditions

# Log-likelihood function
def log_likelihood(theta, t, N, cases):
    beta, gamma = theta
    # Check for valid parameter values
    if beta <= 0 or gamma <= 0 or beta >= 1 or gamma >= 1:
        return -np.inf
    I0 = cases[0] # Initial number of infected individuals
    S0 = N - I0
    R0 = 0
    y0 = [S0, I0, R0]
    try:
        ret = odeint(sir_model, y0, t, args=(beta, gamma, N))
        S, I, R = ret.T
    except:
        return -np.inf


    sigma = 1.0  # Assumed standard deviation of measurement errors
    # Calculate the log-likelihood
    ll = -0.5 * np.sum(((cases - I) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return ll #log(likelyhood) funzione dei parametri 


def log_prior(theta):
    beta, gamma = theta
    # Uniform prior for beta
    if not (0 < beta < 1):
        return -np.inf
    # Uniform prior for gamma
    if not (1/16 < gamma < 1/12):
        return -np.inf
    return 0  # Combine priors for beta and gamma


# Log-posterior function
def log_posterior(theta, t, N, cases):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def switzerland_data():
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
    return dates, cases





def analisi_del_migaele_e_GPT(dates, cases, initial_beta, population,initial_gamma, ndim, nwalkers, nsteps):
    t = np.arange(len(dates))

    # Initial positions of walkers
    pos = [np.array([initial_beta, initial_gamma]) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, population, cases))

    

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

    S_pred, I_pred, R_pred = run_sir_model(t, best_beta, best_gamma, population, cases[0])
    C_pred = I_pred  # Cumulative cases predicted

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
    return beta_mcmc, gamma_mcmc, R0_mcmc, samples


