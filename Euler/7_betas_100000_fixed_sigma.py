# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:13:36 2024

@author: Ninja
"""
import numpy as np
from scipy.integrate import odeint
import emcee
import os
cases=[1.89595044e+01, 2.52793395e+01, 3.15991746e+01, 3.89723157e+01,
       6.00384312e+01, 9.16376058e+01, 2.20140916e+02, 2.78072736e+02,
       3.49697532e+02, 3.93936376e+02, 5.15066545e+02, 6.75169029e+02,
       8.91096722e+02, 1.16811615e+03, 1.41248310e+03, 2.29199346e+03,
       2.28567362e+03, 2.75228810e+03, 3.10303894e+03, 3.97728277e+03,
       4.84204684e+03, 6.12286672e+03, 6.99289732e+03, 8.04830975e+03,
       8.74138498e+03, 9.55559038e+03, 1.03855954e+04, 1.15684578e+04,
       1.24216355e+04, 1.27176144e+04, 1.39162765e+04, 1.41753897e+04,
       1.48168529e+04, 1.50949257e+04, 1.51897232e+04, 1.48768914e+04,
       1.48000000e+04, 1.42817736e+04, 1.40879653e+04, 1.36887624e+04,
       1.34623017e+04, 1.27702797e+04, 1.22899723e+04, 1.15294855e+04,
       1.06246958e+04, 1.01591346e+04, 9.75993171e+03, 8.90780730e+03,
       8.18418621e+03, 7.50059073e+03, 6.98973741e+03, 6.58526797e+03,
       6.12707994e+03, 5.76790266e+03, 5.18121132e+03, 4.59978651e+03,
       4.20374352e+03, 3.95832326e+03, 3.77294144e+03, 3.61073234e+03,
       3.14727778e+03, 3.00402819e+03, 2.76492777e+03, 2.54057363e+03,
       2.27935379e+03, 2.18244965e+03, 2.04867982e+03, 1.88647072e+03,
       1.71583518e+03, 1.61050460e+03, 1.42933599e+03, 1.31031244e+03,
       1.24290086e+03, 1.17548929e+03, 1.05962565e+03, 9.23749203e+02,
       8.52124406e+02, 7.95245892e+02, 7.18354568e+02, 6.48836384e+02,
       6.41463243e+02, 6.29876879e+02, 5.98277705e+02, 5.26652909e+02,
       4.99266958e+02, 4.53974808e+02, 4.23428939e+02, 4.01309517e+02,
       3.66550425e+02, 3.50750838e+02, 3.30738026e+02, 2.87552488e+02,
       2.89659100e+02, 2.88605795e+02, 2.69646289e+02, 2.47526867e+02,
       2.30673974e+02, 2.41207033e+02, 2.43313644e+02, 2.41207033e+02,
       2.38047115e+02, 2.39100421e+02, 2.63326454e+02, 2.61219843e+02,
       2.47526867e+02, 2.62273149e+02, 2.68592983e+02, 2.73859512e+02,
       2.86499183e+02, 3.05458687e+02, 3.02298770e+02, 3.74976871e+02,
       3.53910755e+02, 3.44431003e+02, 3.56017367e+02, 3.62337201e+02,
       3.84456623e+02, 4.04469434e+02, 4.45548361e+02, 4.85573982e+02,
       5.26652909e+02, 5.48772331e+02, 5.98277705e+02, 7.91032669e+02,
       8.07885562e+02, 8.51071100e+02, 9.53241765e+02, 1.02697317e+03,
       1.05646574e+03, 1.09017152e+03, 1.17970252e+03, 1.21762153e+03,
       1.26607359e+03, 1.30715252e+03, 1.33137855e+03, 1.36087112e+03,
       1.36929756e+03, 1.27660665e+03, 1.39141699e+03, 1.34717814e+03,
       1.36087112e+03, 1.39036368e+03, 1.38720376e+03, 1.44408228e+03,
       1.45672195e+03, 1.48726781e+03, 1.53993311e+03, 1.58206534e+03,
       1.60839798e+03, 1.61050460e+03, 1.67580956e+03, 1.73900791e+03,
       1.82011245e+03, 1.86856452e+03, 2.11293147e+03, 2.06131949e+03,
       2.08554552e+03, 2.10871825e+03, 2.15085048e+03, 2.21826205e+03,
       2.22563519e+03, 2.26144759e+03, 2.30568644e+03, 2.34781867e+03,
       2.40575049e+03, 2.49106826e+03, 2.50581454e+03, 2.64379760e+03,
       2.54794677e+03, 2.70699595e+03, 2.76914100e+03, 2.83865918e+03,
       2.97453563e+03, 3.06196001e+03, 3.21468936e+03, 3.33265961e+03,
       3.46326953e+03, 3.51804143e+03, 3.53384102e+03, 3.64549144e+03,
       3.77926127e+03, 3.85509929e+03, 3.98570921e+03, 4.08156004e+03,
       4.12158567e+03, 4.14265178e+03, 4.20585013e+03, 4.42493774e+03,
       4.41545799e+03, 4.55344105e+03, 4.73039643e+03, 4.76620883e+03,
       4.81150098e+03, 4.90524519e+03, 4.95264396e+03, 5.15066545e+03,
       5.24546297e+03, 5.43716463e+03, 5.53617538e+03, 5.64045265e+03,
       5.79212869e+03, 5.85322043e+03, 6.05334854e+03, 5.60569356e+03,
       5.13802578e+03, 6.08810763e+03, 6.12918655e+03, 6.09548077e+03,
       6.20397127e+03, 5.91641878e+03, 5.54986835e+03, 5.05165470e+03,
       5.48035017e+03, 5.38344604e+03, 5.27495554e+03, 5.29602165e+03,
       5.36343322e+03, 5.36343322e+03, 5.36343322e+03, 5.83426093e+03,
       6.27138284e+03, 5.81214150e+03, 7.63962710e+03, 8.93519325e+03,
       8.81195647e+03, 8.81195647e+03, 1.23858231e+04, 1.36666430e+04,
       1.61029393e+04, 1.83570138e+04, 2.10281973e+04, 2.10281973e+04,
       2.12335920e+04, 2.88142340e+04, 3.12463171e+04, 3.71406164e+04,
       4.00108748e+04, 4.53901076e+04, 4.53901076e+04, 4.53901076e+04,
       5.93990750e+04, 6.45844995e+04, 7.03050034e+04, 7.73316065e+04,
       8.37388658e+04, 8.37388658e+04, 8.35334712e+04, 9.73370439e+04,
       1.00583332e+05, 1.05267383e+05, 1.10670842e+05, 1.13585339e+05,
       1.13585339e+05, 1.13585339e+05, 1.13467369e+05, 1.13017608e+05,
       1.13105032e+05, 1.10505473e+05, 1.07919607e+05, 1.07919607e+05,
       1.07919607e+05, 9.83734969e+04, 9.67219133e+04, 9.25529289e+04,
       8.71705362e+04, 8.24980716e+04, 8.24980716e+04, 8.24980716e+04,
       7.45382395e+04, 7.27328733e+04, 6.91769129e+04, 6.66226463e+04,
       6.40736462e+04, 6.40736462e+04, 6.40736462e+04, 5.98330370e+04,
       5.90304180e+04, 5.90778167e+04, 5.70586295e+04, 5.64761513e+04,
       5.64761513e+04]

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

def log_likelihood(theta, t, N, case, sigma):
    lppd=np.zeros(len(cases))
    betas = theta[:len(initial_betas)]  # Estrai i beta
    t_changes = theta[len(initial_betas):len(initial_betas)+len(initial_t_changes)]  # Estrai i t_change
    #sigma=theta[len(initial_betas)+len(initial_t_changes):]
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
    sigmas_t = np.concatenate([np.full(int(t_changes[i]) - int(t_changes[i-1]) if i > 0 else int(t_changes[0]), sigma[i]) 
                               for i in range(len(sigma) - 1)] + [np.full(len(cases) - int(t_changes[-1]), sigma[-1])])
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
    lppd=-(0.5*((cases - I_model)**2 / sigmas_t**2 + np.log(2 * np.pi * sigmas_t**2)))
    # Calcola la log-verosimiglianza
    # Deviazione standard assunta
    ll = np.sum(lppd)
    return ll, lppd

def log_prior(theta, t_max):
    betas = theta[:len(initial_betas)]  # Estrai i beta
    t_changes = theta[len(initial_betas):len(initial_betas)+len(initial_t_changes)]  # Estrai i t_change
    #sigmas=theta[len(initial_betas)+len(initial_t_changes):]
    # Controlla che beta sia in [0, 10]
    if not all(0 < beta < 3 for beta in betas):
        return -np.inf

    # Controlla che i tempi siano ordinati e entro i limiti [0, t_max]
    if not (0 < t_changes[0] < 15)&(30<t_changes[1]<43)&(80<t_changes[2]<160) &(190< t_changes[3] < 210) &(210<t_changes[4]<225)&(240<t_changes[5]<260):
        return -np.inf
    
# =============================================================================
#     if not all(0 < sigma< 5000 for sigma in sigmas):
#         return -np.inf
# =============================================================================

# =============================================================================
#     if not (800<sigma[0]<1200) & (800<sigma[1]<1200) & (240<sigma[2]<360) & (4000<sigma[3]<6000) & (4000<sigma[4]<6000):
#         return -np.inf
# =============================================================================
    return 0

def log_posterior(theta, t, N, cases, sigma):
    lp = log_prior(theta, t.max())
    if not np.isfinite(lp):
        return -np.inf, np.full(len(cases), -np.inf)
    ll, lppd = log_likelihood(theta, t, N, cases, sigma)
    if not np.isfinite(ll):
        return -np.inf, np.full(len(cases), -np.inf)
    return lp + ll, lppd





population = 100000*90

#initial_betas = np.full(len(initial_betas), 1/14)
initial_betas=[1.5/14, 2/14,  1/28, 1/14, 1/28, 2/14, 1/56]



initial_t_changes = [10, 37, 120, 200, 220, 253]


gamma = 1/14

initial_sigma=[300, 300, 300, 500, 1000, 2000, 2000]

ndim = len(initial_betas)+len(initial_t_changes)

nwalkers = 50

nsteps = 100000


#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(cases))
# Initial positions of walkers

initial_pos = np.random.normal(
    loc=np.array(list(initial_betas) + list(initial_t_changes) ),
    scale=np.array([0.1] * len(initial_betas) + [2] * len(initial_t_changes)),
    size=(nwalkers, ndim)
)


# Ensure all initial positions are positive
for i in range(nwalkers):
    for j in range(ndim):
        while initial_pos[i, j] <= 0:
            initial_pos[i, j] = np.random.normal(
                loc=np.array(list(initial_betas) + list(initial_t_changes) )[j],
                scale=np.array(
                    [0.2] * len(initial_betas) + [2] * len(initial_t_changes))[j]
            )

base_filename = "7beta_fixed_sigma"
extension = ".h5"
filename = base_filename + extension

# Check if the file exists and modify the filename if it does
counter = 1
while os.path.isfile(filename):
    filename = f"{base_filename}_{counter}{extension}"
    counter += 1
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, population, cases, initial_sigma), backend=backend)


# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(initial_pos, nsteps, progress=True)
print("Done.")