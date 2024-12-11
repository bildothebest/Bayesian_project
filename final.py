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



def get_maps(chain_1, log_prob_1):

    idx_map_1 = np.argmax(log_prob_1)
    map_params = chain_1[idx_map_1]
    map_log_like=np.max(log_prob_1)
    return map_params, map_log_like


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





def log_likelihood(theta, t, N, case):
    lppd=np.zeros(len(cases))
    betas = theta[:len(initial_betas)]  # Estrai i beta
    t_changes = theta[len(initial_betas):len(initial_betas)+len(initial_t_changes)]  # Estrai i t_change
    sigma=theta[len(initial_betas)+len(initial_t_changes):]
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
    lppd=-(0.5*((cases - I_model)**2 / sigmas_t**2 + np.log(2 * np.pi * sigmas_t**2)))
    # Calcola la log-verosimiglianza
    # Deviazione standard assunta
    ll = np.sum(lppd)
    return ll, lppd




def log_prior(theta, t_max):
    betas = theta[:len(initial_betas)]  # Estrai i beta
    t_changes = theta[len(initial_betas):len(initial_betas)+len(initial_t_changes)]  # Estrai i t_change
    sigmas=theta[len(initial_betas)+len(initial_t_changes):]
    # Controlla che beta sia in [0, 10]
    if not all(0 < beta < 3 for beta in betas):
        return -np.inf

    # Controlla che i tempi siano ordinati e entro i limiti [0, t_max]
    if not (0 < t_changes[0] < 15) &(20<t_changes[1]<30)&(30<t_changes[2]<43)&(80<t_changes[3]<160) &(190< t_changes[4] < 210) &(210<t_changes[5]<225)&(240<t_changes[6]<260):
        return -np.inf
    
    if not all(0 < sigma< 2500 for sigma in sigmas):
        return -np.inf

# =============================================================================
#     if not (800<sigma[0]<1200) & (800<sigma[1]<1200) & (240<sigma[2]<360) & (4000<sigma[3]<6000) & (4000<sigma[4]<6000):
#         return -np.inf
# =============================================================================
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

#initial_betas = np.full(len(initial_betas), 1/14)
initial_betas=[1.5/14, 2/14, 1/14, 1/28, 1/14, 1/28, 2/14, 1/56]



initial_t_changes = [14, 26, 37, 120, 200, 220, 253]


gamma = 1/14

initial_sigma=[300, 300, 300, 300, 500, 1000, 1000, 1000]

ndim = len(initial_betas)+len(initial_t_changes)+len(initial_sigma)

nwalkers = 50

nsteps = 100000


#def analisi_del_migaele_e_GPT(dates, cases, initial_params, population, initial_gamma, ndim, nwalkers, nsteps):
t = np.arange(len(dates))
# Initial positions of walkers

initial_pos = np.random.normal(
    loc=np.array(list(initial_betas) + list(initial_t_changes) + list(initial_sigma)),
    scale=np.array([0.1] * len(initial_betas) + [2] * len(initial_t_changes) + len(initial_sigma)*[50]),
    size=(nwalkers, ndim)
)


# Ensure all initial positions are positive
for i in range(nwalkers):
    for j in range(ndim):
        while initial_pos[i, j] <= 0:
            initial_pos[i, j] = np.random.normal(
                loc=np.array(list(initial_betas) + list(initial_t_changes) +  list(initial_sigma))[j],
                scale=np.array(
                    [0.2] * len(initial_betas) + [2] * len(initial_t_changes) +  len(initial_sigma)*[0.5])[j]
            )

base_filename = "8beta_fixed_sigma_2_2"
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
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, population, cases), backend=backend)


# Run MCMC
print("Running MCMC...")
sampler.run_mcmc(initial_pos, nsteps, progress=True)
print("Done.")
#%%
plt.style.use('tableau-colorblind10')

def plot_del_diocane(samples, samples_3D, log_prob,log_prob_3D, n_betas, n_sigmas, n_t_changes, base_filename, fixed_sigma=False, initial_sigma=None, corner_plot=False):
    
    if corner_plot:
        labels = [f"beta{i+1}" for i in range(n_betas)] + [f"t_change_{i+1}" for i in range(n_betas-1)]+[f"sigma_{i+1}" for i in range(len(initial_betas))]
        fig = corner.corner(samples, labels=labels)
        plt.savefig(f"./Plots/Corner_{base_filename}.png", dpi=600, bbox_inches='tight', format='png')  # Higher DPI for better quality
        plt.show()
    
    
    
    #Model Predictiv
    n_posterior_samples = 200
    positions=np.random.choice(len(samples), n_posterior_samples, replace=False)
    posterior_samples = samples[positions]
    log_prob_samples=log_prob[positions]
    model_predictive = []
    map_params, map_log_like=get_maps(posterior_samples, log_prob_samples)
    times_map = map_params[n_betas:n_betas+n_t_changes]
    for sample in posterior_samples:
        betas = sample[:n_betas]  # Estrai i beta
        times = sample[n_betas:n_betas+n_t_changes]  # Estrai i t_change
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
        model_predictive.append(single_posterior_predictive)
    model_predictive = np.array(model_predictive)
    
    mean_prediction = np.mean(model_predictive, axis=0)
    lower_bound = np.percentile(model_predictive, 2.5, axis=0)
    upper_bound = np.percentile(model_predictive, 97.5, axis=0)
    
    
    ##Plot confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)
    
    plt.plot(t, mean_prediction, label="Mean Prediction", color="blue")
    plt.fill_between(t, lower_bound, upper_bound, color="blue", alpha=0.3, label="95% CI")
    plt.xlabel("Time")
    plt.ylabel("Infected Individuals")
    plt.legend()
    plt.title(f"Model Predictive Plot with $t_c$ file: {base_filename}")
    plt.vlines(x=times_map, ymin=-6000, ymax=130000,color='red', linestyle='dashed', linewidth=1.5)
    plt.savefig(f"./Plots/Model Predictive Trajectories{base_filename}.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
    plt.show()
    
    ##plot single simulations
    plt.figure(figsize=(12, 6))
    for trajectory in model_predictive:
        plt.plot(t, trajectory, color="blue", alpha=0.1)  # Individual trajectories
    
    # Highlight the mean trajectory
    mean_trajectory = np.mean(model_predictive, axis=0)
    plt.plot(t, mean_trajectory, color="red", label="Mean Trajectory", linewidth=2)
    plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)
    
    # Add labels, legend, and title
    plt.xlabel("Time")
    plt.ylabel("Infected Individuals")
    plt.title(f"Model Predictive Trajectories file: {base_filename}")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    map_params, map_log=get_maps(samples, log_prob)
    print(map_params)
    map_predictive = []
    for sample in [map_params]:
        betas = sample[:n_betas]  # Estrai i beta
        times = sample[n_betas:n_betas+n_t_changes]  # Estrai i t_change
        if not fixed_sigma:
            initial_sigma=sample[n_betas+n_t_changes:]
            print('in the loop', initial_sigma)
            print(sample)
        print(initial_sigma)
        R0=0
        I0=cases[0]
        times=np.concatenate(([t[0]], times, [t[-1]]))
        single_posterior_predictive=[]
        sigmas_single=[]
        for i in range(n_betas):
            if i == n_betas - 1:  # Ultima iterazione
                t_segment = t[(times[i] <= t) & (t <= times[i + 1])]  # Usa <= per includere l'ultimo punto
            else:
                t_segment = t[(times[i] <= t) & (t < times[i + 1])]  # Condizione originale per tutte le altre iterazioni
            S, I, R=run_sir_model(t_segment, betas[i], gamma, population, I0, R0)
            I0=I[-1]
            R0=R[-1]
            sigmas_temp=np.full(len(t_segment), initial_sigma[i])
            sigmas_single.append(sigmas_temp)
            single_posterior_predictive.append(I)
        single_posterior_predictive = np.concatenate(single_posterior_predictive)
        sigmas_single=np.concatenate(sigmas_single)
        map_predictive.append(single_posterior_predictive)
    map_predictive = np.array(map_predictive)
    map_predictive=np.reshape(map_predictive, -1)
    ##Plot confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(t, cases, label="Observed Data", color="black", alpha=0.7)
    
    plt.plot(t, map_predictive, label="MAP Prediction", color="blue")
    plt.fill_between(t, map_predictive-sigmas_single, map_predictive+sigmas_single, color="blue", alpha=0.3, label="95% CI")
    plt.title(f"MAP and std file: {base_filename}")
    plt.show()
    
    #Plot posterior predictiv
    n_posterior_samples = 200
    posterior_samples = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]
    
    model_predictive=[]
    posterior_predictive = []
    for sample in posterior_samples:
        betas = sample[:n_betas]  # Estrai i beta
        times = sample[n_betas:n_betas+n_t_changes]  # Estrai i t_change
        if fixed_sigma==False:
            initial_sigma=sample[n_betas+n_t_changes:]
        R0=0
        I0=cases[0]
        times=np.concatenate(([t[0]], times, [t[-1]]))
        single_model_predictive=[]
        sigmas_single=[]
        for i in range(n_betas):
            if i == n_betas - 1:  # Ultima iterazione
                t_segment = t[(times[i] <= t) & (t <= times[i + 1])]  # Usa <= per includere l'ultimo punto
            else:
                t_segment = t[(times[i] <= t) & (t < times[i + 1])]  # Condizione originale per tutte le altre iterazioni
            S, I, R=run_sir_model(t_segment, betas[i], gamma, population, I0, R0)
            I0=I[-1]
            R0=R[-1]
            sigmas_temp=np.full(len(t_segment), initial_sigma[i])
            sigmas_single.append(sigmas_temp)
            single_model_predictive.append(I)
        single_model_predictive = np.concatenate(single_model_predictive)
        sigmas_single=np.concatenate(sigmas_single)
        model_predictive.append(single_model_predictive)
        single_posterior_predictive=np.random.normal(loc=single_model_predictive, scale=sigmas_single)
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
    plt.title(f"Posterior Predictive Plot with $t_c$ file: {base_filename}")
    plt.vlines(x=times_map, ymin=-6000, ymax=130000,color='red', linestyle='dashed', linewidth=1.5)
    plt.savefig(f"./Plots/Posterior Predictive {base_filename}.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
    plt.show()
    
    ##plot single simulations
    plt.figure(figsize=(12, 6))
    for trajectory in posterior_predictive:
        plt.plot(t, trajectory, color='pink',alpha=0.05)  # Individual trajectories
    
    # Highlight the mean trajectory
    mean_trajectory = np.mean(posterior_predictive, axis=0)
    plt.plot(t, mean_trajectory,color='darkred', label="Mean Trajectory", linewidth=2)
    plt.plot(t, cases, label="Observed Data",color='seagreen')
    
    # Add labels, legend, and title
    plt.xlabel("Time")
    plt.ylabel("Infected Individuals")
    plt.title(f"Posterior Predictive Trajectories with $t_c$ file: {base_filename}")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # =============================================================================
    # Traces
    # =============================================================================
    # Extract the samples for each parameter after burn-in (discard the first 200 steps)
    # Trace plot for each parameter
    if not fixed_sigma:
        n_plots=n_betas+n_t_changes+n_sigmas+1
        print(n_plots)
    else:
        n_plots=n_betas+n_t_changes+1
    fig, axes = plt.subplots(n_plots, figsize=(15, 25), sharex=True)
    labels = [f"beta{i+1}" for i in range(n_betas)]+ [f"t_change{i+1}" for i in range(n_t_changes)] + ([f"sigma{i+1}" for i in range(n_sigmas)] if not fixed_sigma else [])
    print(labels)
    for i in range(n_plots-1):
        ax = axes[i]
        ax.plot(samples_3D[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, samples_3D.shape[0])
        print(samples_3D.shape[0])
        ax.set_ylabel(labels[i])
        ax.grid()
    
    # Plot the log-likelihood trace
    axes[n_plots-1].plot(log_prob_3D, "k", alpha=0.3)
    axes[n_plots-1].set_ylabel("Log-likelihood")
    axes[n_plots-1].set_xlabel("Step number")
    axes[n_plots-1].grid()
    
    plt.tight_layout()
    plt.savefig(f"./Plots/{base_filename}.png", dpi=900, bbox_inches='tight', format='png')  # Higher DPI for better quality
    plt.show()
    #return beta_mcmc, gamma_mcmc, R0_mcmc, samples




#%%
# Get the samples
base_filename1='8beta_fixed_sigma'

n_betas1=8
n_sigmas1=8
n_t_changes1=7


filename = f"./Euler/{base_filename1}.h5"
sampler1=emcee.backends.HDFBackend(filename)
tau1=[12119.86598155, 13873.26186916, 12236.35427208,  9810.62077337,
 12647.91082375, 13233.91827366,11689.0396601, 4983.90122503,
  6734.59010157,  7342.310691,    9048.07123193, 10747.13623488,
 13007.57960977, 10464.09851998,  8533.82581491,  5331.32597187,
  5372.53087715,  5868.04783611,  9847.86934522, 12784.29473094,
  6106.63357579, 13174.64385446, 13375.41097887]
max_autocorr1 = max(tau1)
discard1=int(max_autocorr1*5)

thin=10

#Corner no R0
samples1 = sampler1.get_chain(discard=discard1, thin=thin, flat=True)
samples1_3D = sampler1.get_chain(discard=discard1, thin=thin, flat=False)

log_prob_1_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=True)
log_prob_1 = log_prob_1_dirty[log_prob_1_dirty != -np.inf]

log_prob_1_3D_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=False)
slices_with_neg_inf1 = np.any(np.isinf(log_prob_1_3D_dirty), axis=0)
log_prob_1_3D = log_prob_1_3D_dirty[:, ~slices_with_neg_inf1]

#%%
base_filename2='7beta_inferr_sigma'
n_betas2=7
n_sigmas2=7
n_t_changes2=6
# Get the samples
filename = f"./Euler/{base_filename2}.h5"
sampler2=emcee.backends.HDFBackend(filename)

max_autocorr2 = 14094.57812518
discard2=int(max_autocorr2*5)

thin=10

#Corner no R0
samples2 = sampler2.get_chain(discard=discard2, thin=thin, flat=True)
samples2_3D= sampler2.get_chain(discard=discard2, thin=thin, flat=False)

log_prob_2_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=True)
log_prob_2 = log_prob_2_dirty[log_prob_2_dirty != -np.inf]

log_prob_2_3D_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=False)
slices_with_neg_inf2 = np.any(np.isinf(log_prob_2_3D_dirty), axis=0)
log_prob_2_3D = log_prob_2_3D_dirty[:, ~slices_with_neg_inf2]

#%%
base_filename3='7beta_fixed_sigma'
n_betas3=7
n_sigmas3=0
initial_sigma3=[300, 300, 300, 500, 1000, 2000, 2000]
n_t_changes3=6
# Get the samples
filename = f"./Euler/{base_filename3}.h5"
sampler3=emcee.backends.HDFBackend(filename)
max_autocorr3 = 12025.99091214
discard3=int(max_autocorr3*5)
thin=10

#Corner no R0
samples3 = sampler3.get_chain(discard=discard3, thin=thin, flat=True)
samples3_3D= sampler3.get_chain(discard=discard3, thin=thin, flat=False)

log_prob_3_dirty=sampler3.get_log_prob(discard=discard3, thin=thin, flat=True)
log_prob_3 = log_prob_3_dirty[log_prob_3_dirty != -np.inf]

log_prob_3_3D_dirty=sampler3.get_log_prob(discard=discard3, thin=thin, flat=False)
slices_with_neg_inf = np.any(np.isinf(log_prob_3_3D_dirty), axis=0)
log_prob_3_3D = log_prob_3_3D_dirty[:, ~slices_with_neg_inf]

#%%
base_filename4='6beta_inferr_sigma'
n_betas4=6
n_sigmas4=6
n_t_changes4=5
# Get the samples
filename = f"./Euler/{base_filename4}.h5"
sampler4=emcee.backends.HDFBackend(filename)

max_autocorr4 = 7737.15967022
discard4=int(max_autocorr4*5)
thin=10

#Corner no R0
samples4 = sampler4.get_chain(discard=discard4, thin=thin, flat=True)
samples4_3D= sampler4.get_chain(discard=discard4, thin=thin, flat=False)

log_prob_4_dirty=sampler4.get_log_prob(discard=discard4, thin=thin, flat=True)
log_prob_4 = log_prob_4_dirty[log_prob_4_dirty != -np.inf]

log_prob_4_3D_dirty=sampler4.get_log_prob(discard=discard4, thin=thin, flat=False)
slices_with_neg_inf4 = np.any(np.isinf(log_prob_4_3D_dirty), axis=0)
log_prob_4_3D = log_prob_4_3D_dirty[:, ~slices_with_neg_inf4]

#%%
base_filename5='6beta_fixed_sigma'
n_betas5=6
n_sigmas5=0
n_t_changes5=5
# Get the samples
filename = f"./Euler/{base_filename5}.h5"
sampler5=emcee.backends.HDFBackend(filename)
initial_sigma5=[300, 300, 300, 500, 2000, 2000]

max_autocorr5 = 9822.48244725
discard5=int(max_autocorr5*5)
thin=10

#Corner no R0
samples5 = sampler5.get_chain(discard=discard5, thin=thin, flat=True)
samples5_3D= sampler5.get_chain(discard=discard5, thin=thin, flat=False)

log_prob_5_dirty=sampler5.get_log_prob(discard=discard5, thin=thin, flat=True)
log_prob_5 = log_prob_5_dirty[log_prob_5_dirty != -np.inf]

log_prob_5_3D_dirty=sampler5.get_log_prob(discard=discard5, thin=thin, flat=False)
slices_with_neg_inf = np.any(np.isinf(log_prob_5_3D_dirty), axis=0)
log_prob_5_3D = log_prob_5_3D_dirty[:, ~slices_with_neg_inf]


#%%
base_filename6='5beta_inferr_sigma'
n_betas6=5
n_sigmas6=5
n_t_changes6=4
# Get the samples
filename = f"./Euler/{base_filename6}.h5"
sampler6=emcee.backends.HDFBackend(filename)

max_autocorr6 = 13507.16344972
discard6=int(max_autocorr6*5)
thin=10

#Corner no R0
samples6 = sampler6.get_chain(discard=discard6, thin=thin, flat=True)
samples6_3D= sampler6.get_chain(discard=discard6, thin=thin, flat=False)

log_prob_6_dirty=sampler6.get_log_prob(discard=discard6, thin=thin, flat=True)
log_prob_6 = log_prob_6_dirty[log_prob_6_dirty != -np.inf]

log_prob_6_3D_dirty=sampler6.get_log_prob(discard=discard6, thin=thin, flat=False)
slices_with_neg_inf6 = np.any(np.isinf(log_prob_6_3D_dirty), axis=0)
log_prob_6_3D = log_prob_6_3D_dirty[:, ~slices_with_neg_inf6]

#%%
base_filename7='5beta_fixed_sigma'
n_betas7=5
n_sigmas7=0
n_t_changes7=4

initial_sigma7=[500, 500, 300, 2000, 2000]

# Get the samples
filename = f"./Euler/{base_filename7}.h5"
sampler7=emcee.backends.HDFBackend(filename)

max_autocorr7 = 5600.43517761
discard7=int(max_autocorr7*5)
thin=10

#Corner no R0
samples7 = sampler7.get_chain(discard=discard7, thin=thin, flat=True)
samples7_3D= sampler7.get_chain(discard=discard7, thin=thin, flat=False)

log_prob_7_dirty=sampler7.get_log_prob(discard=discard7, thin=thin, flat=True)
log_prob_7 = log_prob_7_dirty[log_prob_7_dirty != -np.inf]

log_prob_7_3D_dirty=sampler7.get_log_prob(discard=discard7, thin=thin, flat=False)
slices_with_neg_inf = np.any(np.isinf(log_prob_7_3D_dirty), axis=0)
log_prob_7_3D = log_prob_7_3D_dirty[:, ~slices_with_neg_inf]

#%%

#3rd
plot_del_diocane(samples1,samples1_3D, log_prob_1,log_prob_1_3D, n_betas1, n_sigmas1, n_t_changes1, base_filename1, fixed_sigma=False, initial_sigma=None, corner_plot=False)

#Best
plot_del_diocane(samples2,samples2_3D, log_prob_2,log_prob_2_3D, n_betas2, n_sigmas2, n_t_changes2, base_filename2, fixed_sigma=False, initial_sigma=None, corner_plot=False)

# =============================================================================
# plot_del_diocane(samples3,samples3_3D, log_prob_3,log_prob_3_3D, n_betas3, n_sigmas3, n_t_changes3, base_filename3, fixed_sigma=True, initial_sigma=initial_sigma3, corner=False)
# =============================================================================
#2nd
plot_del_diocane(samples4,samples4_3D, log_prob_4,log_prob_4_3D, n_betas4, n_sigmas4, n_t_changes4, base_filename4, fixed_sigma=False, initial_sigma=None, corner_plot=True)

# =============================================================================
# plot_del_diocane(samples5,samples5_3D, log_prob_5,log_prob_5_3D, n_betas5, n_sigmas5, n_t_changes5, base_filename5, fixed_sigma=True, initial_sigma=initial_sigma5, corner=False)
# =============================================================================

# WORSE=============================================================================
# plot_del_diocane(samples6,samples6_3D, log_prob_6,log_prob_6_3D, n_betas6, n_sigmas6, n_t_changes6, base_filename6, fixed_sigma=False, initial_sigma=None, corner_plot=True)
# =============================================================================

# =============================================================================
# plot_del_diocane(samples7,samples7_3D, log_prob_7,log_prob_7_3D, n_betas7, n_sigmas7, n_t_changes7, base_filename7, fixed_sigma=True, initial_sigma=initial_sigma7, corner=False)
# =============================================================================

#%%
def chi_squared(cases, sigmas, mu):
    return np.sum((cases - mu)**2/sigmas**2)


def test_statistic(cases, sigma, model_predictive):
    mu = model_predictive
    chi = chi_squared(cases, sigma, mu)
    return chi

def ppd_model_check(test_statistic, sigmas, cases, ppd, model_predictive):
    t_data = []
    t_rep = []
    for y_rep, single_mod_pred, sigma in zip(ppd, model_predictive, sigmas):
        t_data.append(test_statistic(cases, sigma, single_mod_pred))
        t_rep.append(test_statistic(y_rep, sigma, single_mod_pred))

    t_data = np.array(t_data)
    t_rep = np.array(t_rep)

    pte = (t_rep >= t_data).sum()/len(t_data)
    return pte, t_rep, t_data


def get_ppd(fixed_sigma, n_t_changes, n_betas, n_posterior_samples, samples, initial_sigma=None):

    ppd_params = samples[np.random.choice(len(samples), n_posterior_samples, replace=False)]

    model_predictive=[]
    posterior_predictive = []
    sigma_single=[]
    for sample in ppd_params:
        betas = sample[:n_betas]  # Estrai i beta
        times = sample[n_betas:n_betas+ n_t_changes]  # Estrai i t_change

        if fixed_sigma==False:
            initial_sigma=sample[n_betas+ n_t_changes:]
        R0=0
        I0=cases[0]
        times=np.concatenate(([t[0]], times, [t[-1]]))
        single_model_predictive=[]
        sigmas_single_temp=[]

        for i in range(n_betas):
            if i == n_betas - 1:  # Ultima iterazione
                t_segment = t[(times[i] <= t) & (t <= times[i + 1])]  # Usa <= per includere l'ultimo punto
            else:
                t_segment = t[(times[i] <= t) & (t < times[i + 1])]  # Condizione originale per tutte le altre iterazioni
            S, I, R=run_sir_model(t_segment, betas[i], gamma, population, I0, R0)
            I0=I[-1]
            R0=R[-1]
            sigmas_temp=np.full(len(t_segment), initial_sigma[i])
            sigmas_single_temp.append(sigmas_temp)
            single_model_predictive.append(I)
        
        sigmas_single_temp=np.concatenate(sigmas_single_temp)
        single_model_predictive = np.concatenate(single_model_predictive)
        sigma_single.append(sigmas_single_temp)
        model_predictive.append(single_model_predictive)
        single_posterior_predictive=np.random.normal(loc=single_model_predictive, scale=sigmas_single_temp)
        posterior_predictive.append(single_posterior_predictive)
            
    ppd = np.array(posterior_predictive)
    model_predictive = np.array(model_predictive)
    sigmas = np.array(sigma_single)

    return ppd, model_predictive, sigmas


ppd1, model_predictive1, sigmas1 = get_ppd(False, n_t_changes1, n_betas1, 500, samples1)
pte1, t_rep1, t_data1 = ppd_model_check(test_statistic, sigmas1, cases, ppd1, model_predictive1)
print(pte1)

ppd2, model_predictive2, sigmas2 = get_ppd(False, n_t_changes2, n_betas2, 500, samples2)
pte2, t_rep2, t_data2 = ppd_model_check(test_statistic, sigmas2, cases, ppd2, model_predictive2)
print(pte2)

# =============================================================================
# 
# ppd3, model_predictive3, sigmas3 = get_ppd(True, n_t_changes3, n_betas3, 500, samples3, initial_sigma=initial_sigma3)
# pte3, t_rep3, t_data3 = ppd_model_check(test_statistic, sigmas3, cases, ppd3, model_predictive3)
# print(pte3)
# =============================================================================

ppd4, model_predictive4, sigmas4 = get_ppd(False, n_t_changes4, n_betas4, 500, samples4)
pte4, t_rep4, t_data4 = ppd_model_check(test_statistic, sigmas4, cases, ppd4, model_predictive4)
print(pte4)

# =============================================================================
# ppd5, model_predictive5, sigmas5 = get_ppd(True, n_t_changes5, n_betas5, 500, samples5, initial_sigma=initial_sigma5)
# pte5, t_rep5, t_data5 = ppd_model_check(test_statistic, sigmas5, cases, ppd5, model_predictive5)
# print(pte5)
# =============================================================================

ppd6, model_predictive6, sigmas6 = get_ppd(False, n_t_changes6, n_betas6, 500, samples6)
pte6, t_rep6, t_data6 = ppd_model_check(test_statistic, sigmas6, cases, ppd6, model_predictive6)
print(pte6)

# =============================================================================
# ppd7, model_predictive7, sigmas7 = get_ppd(True, n_t_changes7, n_betas7, 500, samples7, initial_sigma=initial_sigma7)
# pte7, t_rep7, t_data7 = ppd_model_check(test_statistic, sigmas7, cases, ppd7, model_predictive7)
# print(pte7)
# =============================================================================

#%%#WAIC, DIC and bayes ratio jaco
def waic(sampler, discard):
    blobs=sampler.get_blobs(discard=discard, thin=thin, flat=False)
    slices_with_neg_inf = np.any(np.isinf(blobs), axis=(0, 2))
    timewise_log_likelihood = blobs[:, ~slices_with_neg_inf, :]
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
    return(waic)

def bayes_ratio(chain_1, log_prob_1, chain_2, log_prob_2):

    _, log_prob_1= get_maps(chain_1, log_prob_1)
    _, log_prob_2= get_maps(chain_2, log_prob_2)


    return log_prob_1, log_prob_2

def DIC_comparison(chain_1, log_prob_1):

    _, map_prob_1= get_maps(chain_1, log_prob_1)

    mean_log_py_1 = np.mean(log_prob_1)

    p_D = 2 * (map_prob_1 - mean_log_py_1)
    
    p_V = 2*np.var(log_prob_1)

    DIC = -2 * (map_prob_1 - p_D)


    return DIC, p_V, p_D



waic1=waic(sampler1, discard1)
waic2=waic(sampler2, discard2)

# =============================================================================
# waic3=waic(sampler3, discard3)
# =============================================================================
waic4=waic(sampler4, discard4)

# =============================================================================
# waic5=waic(sampler5, discard5)
# =============================================================================
waic6=waic(sampler6, discard6)

# =============================================================================
# waic7=waic(sampler7, discard7)
# =============================================================================

print('waic1:',waic1,'waic2:', waic2, 'waic1/waic2:',waic1/waic2)
print('waic1:',waic1,'waic4:', waic4, 'waic1/waic4:',waic1/waic4)
print('waic1:',waic1,'waic6:', waic6, 'waic1/waic6:',waic1/waic6)
print('waic2:',waic2,'waic4:', waic4, 'waic2/waic4:',waic2/waic4)
print('waic2:',waic2,'waic6:', waic6, 'waic2/waic6:',waic2/waic6)
print('waic4:',waic4,'waic6:', waic6, 'waic4/waic6:',waic4/waic6)


log_prob_1_b, log_prob_2_b=bayes_ratio(samples1, log_prob_1, samples2, log_prob_2)
log_prob_4_b, log_prob_6_b=bayes_ratio(samples4, log_prob_4, samples6, log_prob_6)

print('Log_prob1',log_prob_1_b,'Log_prob2:', log_prob_2_b, 'Bayes_ratio1/2:',log_prob_1_b/log_prob_2_b)
print('Log_prob1',log_prob_1_b,'Log_prob4:', log_prob_4_b, 'Bayes_ratio1/4:',log_prob_1_b/log_prob_4_b)
print('Log_prob1',log_prob_1_b,'Log_prob6:', log_prob_6_b, 'Bayes_ratio1/6:',log_prob_1_b/log_prob_6_b)
print('Log_prob2',log_prob_2_b,'Log_prob4:', log_prob_4_b, 'Bayes_ratio2/4:',log_prob_2_b/log_prob_4_b)
print('Log_prob2',log_prob_2_b,'Log_prob6:', log_prob_6_b, 'Bayes_ratio2/6:',log_prob_2_b/log_prob_6_b)
print('Log_prob4',log_prob_4_b,'Log_prob6:', log_prob_6_b, 'Bayes_ratio4/6:',log_prob_4_b/log_prob_6_b)


DIC_1, p_V_1, p_D_1=DIC_comparison(samples1, log_prob_1)
DIC_2, p_V_2, p_D_2=DIC_comparison(samples2, log_prob_2)
DIC_4, p_V_4, p_D_4=DIC_comparison(samples4, log_prob_4)
DIC_6, p_V_6, p_D_6=DIC_comparison(samples6, log_prob_6)


print('DIC1:',DIC_1, 'p_V_1',p_V_1 ,'p_D_1',p_D_1 ,'DIC2:', DIC_2,'p_V_2:', p_V_2,'p_D_2:', p_D_2 ,'DIC1/DIC2:',DIC_1/DIC_2)
print('DIC1:',DIC_1, 'p_V_1',p_V_1 ,'p_D_1',p_D_1 ,'DIC2:', DIC_4,'p_V_4:', p_V_4,'p_D_4:', p_D_4 ,'DIC1/DIC2:',DIC_1/DIC_4)
print('DIC1:',DIC_1, 'p_V_1',p_V_1 ,'p_D_1',p_D_1 ,'DIC2:', DIC_6,'p_V_6:', p_V_6,'p_D_6:', p_D_6 ,'DIC1/DIC2:',DIC_1/DIC_6)
print('DIC2:',DIC_2, 'p_V_2',p_V_2 ,'p_D_2',p_D_2 ,'DIC2:', DIC_4,'p_V_4:', p_V_4,'p_D_4:', p_D_4 ,'DIC1/DIC2:',DIC_2/DIC_4)
print('DIC2:',DIC_2, 'p_V_2',p_V_2 ,'p_D_2',p_D_2 ,'DIC2:', DIC_6,'p_V_6:', p_V_6,'p_D_6:', p_D_6 ,'DIC1/DIC2:',DIC_2/DIC_6)
print('DIC4:',DIC_4, 'p_V_4',p_V_4 ,'p_D_4',p_D_2 ,'DIC2:', DIC_6,'p_V_6:', p_V_6,'p_D_6:', p_D_6 ,'DIC1/DIC2:',DIC_4/DIC_6)




#%%  Calculate R0 and do corner 
import arviz
import xarray as xr

samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

print(f"Number of samples: {samples.shape[0]}")

R0_list = [samples1[:, i] / gamma for i in range(n_betas1)]
samples1_R0 = np.column_stack((samples1, *R0_list))

labels={'var': [f"beta{i+1}" for i in range(n_betas1)] + [f"t_change_{i+1}" for i in range(n_t_changes1)]+[f"sigma_{i+1}" for i in range(n_sigmas1)]+[f"R0_{i+1}" for i in range(n_betas1)]}
data_xr1 = xr.DataArray(samples1_R0, dims=["draw", "var"], coords=labels)

arviz.plot_posterior(data_xr1)


R0_list2 = [samples2[:, i] / gamma for i in range(n_betas2)]
samples2_R0 = np.column_stack((samples2, *R0_list2))
np.round(np.mean(samples2_R0, axis=0), 2)
labels2={'var': [f"beta{i+1}" for i in range(n_betas2)] + [f"t_change_{i+1}" for i in range(n_t_changes2)]+[f"sigma_{i+1}" for i in range(n_sigmas2)]+[f"R0_{i+1}" for i in range(n_betas2)]}
data_xr2 = xr.DataArray(samples2_R0, dims=["draw", "var"], coords=labels2)


arviz.plot_posterior(data_xr2)



R0_list4 = [samples4[:, i] / gamma for i in range(n_betas4)]
samples4_R0 = np.column_stack((samples4, *R0_list4))
np.round(np.mean(samples4_R0, axis=0), 2)

labels4={'var': [f"beta{i+1}" for i in range(n_betas4)] + [f"t_change_{i+1}" for i in range(n_t_changes4)]+[f"sigma_{i+1}" for i in range(n_sigmas4)]+[f"R0_{i+1}" for i in range(n_betas4)]}
data_xr4 = xr.DataArray(samples4_R0, dims=["draw", "var"], coords=labels4)

arviz.plot_posterior(data_xr4)


# Crea il grafico corner
fig = corner.corner(samples, labels=labels)
plt.show()


# Calcola le medie e deviazioni standard
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)

# Stampa i risultati per i beta
for i in range(len(initial_betas)):
    print(f"Estimated beta{i+1}: {means[i]:.4f} ± {stds[i]:.4f}")

# Stampa i risultati per t_change
for i in range(len(initial_betas)-1):
    idx = len(initial_betas) + i
    print(f"Estimated t_change_{i+1}: {means[idx]:.4f} ± {stds[idx]:.4f}")

# Stampa i risultati per R0
for i in range(len(initial_betas)):
    idx = len(initial_betas) + (len(initial_betas)-1) + i
    print(f"Estimated R0_{i+1}: {means[idx]:.4f} ± {stds[idx]:.4f}")



