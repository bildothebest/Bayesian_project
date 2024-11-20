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



print(beta_gamma_R0)





