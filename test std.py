# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:11:14 2024

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
import matplotlib.pyplot as plt
from IPython import get_ipython  # For IPython utilities
from scipy.optimize import curve_fit


get_ipython().run_line_magic('matplotlib', 'inline')
dates, cases=switzerland_data()
cases=cases[0:281]
cases=cases*90
timepoints=np.linspace(1, len(cases), len(cases))
plt.plot(timepoints, cases)
plt.show()

plt.plot(timepoints[90:180], cases[90:180])
plt.show()
from scipy.stats import linregress

# Parameters
window_size = 10
num_points = len(cases)
sigma_estimates = []

# Sliding window to fit a line and infer sigma
for i in range(90, 180):
    # Get the current window
    x_window = timepoints[i:i + window_size]
    y_window = cases[i:i + window_size]
    
    # Fit a line (linear regression)
    slope, intercept, _, _, _ = linregress(x_window, y_window)
    
    # Calculate residuals (differences between actual and fitted values)
    fitted_values = slope * x_window + intercept
    residuals = y_window - fitted_values
    
    # Estimate sigma as the standard deviation of the residuals
    sigma = np.std(residuals)
    sigma_estimates.append(sigma)

# Calculate the mean sigma over all sliding windows
mean_sigma_lin = np.mean(sigma_estimates)

min_sigma_lin=sigma_estimates[np.argmin(sigma_estimates)]

mean_sigma_lin

#%%
# Define the exponential function
def exponential_function(x, a, b):
    return a * np.exp(b * x)


# Sliding window to fit an exponential function
sigma_estimates = []
for i in range(90, 180):
    # Get the current window
    x_window = timepoints[i:i + window_size]
    y_window = cases[i:i + window_size]
    
    # Fit the exponential function to the data
    try:
        popt, _ = curve_fit(exponential_function, x_window, y_window, p0=(1, 0.01))
        a, b = popt  # Extract fitted parameters
        
        # Calculate residuals (differences between actual and fitted values)
        fitted_values = exponential_function(x_window, a, b)
        residuals = y_window - fitted_values
        
        # Estimate sigma as the standard deviation of the residuals
        sigma = np.std(residuals)
        sigma_estimates.append(sigma)
    except RuntimeError:
        # Handle cases where the fit fails
        sigma_estimates.append(np.nan)

# Calculate the mean sigma over all sliding windows, ignoring NaN values
mean_sigma_exp = np.nanmean(sigma_estimates)

# Display the results
mean_sigma_exp

minsigma_exp=sigma_estimates[np.nanargmin(sigma_estimates)]
