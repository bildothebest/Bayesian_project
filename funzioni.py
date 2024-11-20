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
