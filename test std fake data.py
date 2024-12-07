# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:39:35 2024

@author: Ninja
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
time_steps = 200
sigma = 1.0  # Standard deviation for the noise

# Generate time values
time = np.arange(1, time_steps + 1)

# Generate data centered around exp(time) with added noise
while True:
    data = np.exp(0.03 * time) + np.random.normal(0, sigma, size=time_steps)
    if np.all(data > 0):
        break
# Display the first 10 values of the generated data
data[:10]  # Showing a sample of the data

plt.plot(time, data)
from scipy.stats import linregress

# Parameters
window_size = 10
num_points = len(data)
sigma_estimates = []

# Sliding window to fit a line and infer sigma
for i in range(num_points - window_size + 1):
    # Get the current window
    x_window = time[i:i + window_size]
    y_window = data[i:i + window_size]
    
    # Fit a line (linear regression)
    slope, intercept, _, _, _ = linregress(x_window, y_window)
    
    # Calculate residuals (differences between actual and fitted values)
    fitted_values = slope * x_window + intercept
    residuals = y_window - fitted_values
    
    # Estimate sigma as the standard deviation of the residuals
    sigma = np.std(residuals)
    sigma_estimates.append(sigma)

# Calculate the mean sigma over all sliding windows
mean_sigma = np.mean(sigma_estimates)

# Display the results
mean_sigma


# Apply log transformation to the data
log_data = np.log(data)

mu_log = np.mean(log_data)
sigma_log = np.std(log_data)  # This is the inferred sigma in log space

# Compute the original sigma in linear space
sigma_linear = np.exp(mu_log) * np.sqrt(np.exp(sigma_log**2) - 1)


