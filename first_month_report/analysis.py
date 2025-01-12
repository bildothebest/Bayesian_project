import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import norm
import emcee
import corner
import os
from IPython import get_ipython  # For IPython utilities

get_ipython().run_line_magic('matplotlib', 'inline')


cases_all=[1.89595044e+01, 2.52793395e+01, 3.15991746e+01, 3.89723157e+01,
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
cases=cases_all[:30]
t = np.arange(len(cases))
gamma = 1/14
population = 100000*90

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

def log_likelihood(theta, t, N, cases):
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
        return -np.inf

    # Combina i segmenti per ottenere l'intero modello
    I_model = np.concatenate(I_model_segments)
    # Calcola la log-verosimiglianza
    # Deviazione standard assunta
    ll = np.sum(-(0.5*((cases - I_model)**2 / sigmas_t**2 + np.log(2 * np.pi * sigmas_t**2))))
    return ll

def log_prior(theta):
    betas = theta[:len(initial_betas)]  # Estrai i beta
    t_changes = theta[len(initial_betas):len(initial_betas)+len(initial_t_changes)]  # Estrai i t_change
    sigmas=theta[len(initial_betas)+len(initial_t_changes):]
    # Controlla che beta sia in [0, 10]
    if not all(0 < beta < 3 for beta in betas):
        return -np.inf

    # Controlla che i tempi siano ordinati e entro i limiti [0, t_max]
    if not (all(t_changes[i] <= t_changes[i + 1] for i in range(len(t_changes) - 1)) & (t_changes[0]>0) & (t_changes[-1]<len(cases))):
        return -np.inf
    
    if not all(0 < sigma< 10000 for sigma in sigmas):
        return -np.inf

# =============================================================================
#     if not (800<sigma[0]<1200) & (800<sigma[1]<1200) & (240<sigma[2]<360) & (4000<sigma[3]<6000) & (4000<sigma[4]<6000):
#         return -np.inf
# =============================================================================
    return 0


def log_posterior(theta, t, N, cases):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, t, N, cases)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll



#%%Plot cases
##Plot confidence interval
midpoint = len(t) // 2  # Find the midpoint
first_half_t = t[:midpoint+1]
first_half_cases = cases[:midpoint+1]

second_half_t = t[midpoint:]
second_half_cases = cases[midpoint:]

plt.figure(figsize=(10, 6))

plt.plot(first_half_t, first_half_cases, color='red', linewidth=3, label='First Half')

plt.plot(second_half_t, second_half_cases, color='blue', linewidth=3, label='Second Half')
plt.xlabel("Time (days)", fontsize=16)
plt.ylabel("Infected Individuals", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
params = {'mathtext.default': 'regular'}          
plt.rcParams.update(params)
plt.vlines(x=15, ymin=-1000, ymax=13000,color='red', linestyle='dashed', linewidth=2.5, label='$t_{change}$')
plt.legend(fontsize=16, loc='upper left')  
plt.savefig(f"Cases.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
plt.show()
#%%
plt.style.use('tableau-colorblind10')

def to_subscript(num):
    subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(num).translate(subscript_digits)


def plot_del_diocane(samples, samples_corner,samples_traces, samples_3D, log_prob,log_prob_3D,log_prob_3D_traces, n_betas, 
                     n_sigmas, n_t_changes, base_filename, fixed_sigma=False, initial_sigma=None, corner_plot=False, 
                     model_predictive=False, posterior_predictive=False, traces=False
):
    
    if corner_plot:
        label_kwargs=dict(fontsize=21)
        params = {'mathtext.default': 'regular', 'xtick.labelsize': 14, 'ytick.labelsize': 14}          
        plt.rcParams.update(params)
        if n_betas==1:
            labels = [f"\u03B2" for i in range(n_betas)] + ["$t_{change}$" for i in range(n_betas-1)] + [f"\u03C3" for i in range(n_betas)]        
        else:
            labels = [f"\u03B2{to_subscript(i+1)}" for i in range(n_betas)] + ["$t_{change}$" for i in range(n_betas-1)] + [f"\u03C3{to_subscript(i+1)}" for i in range(n_betas)]        
        fig = corner.corner(samples_corner, labels=labels, label_kwargs=label_kwargs)
        plt.savefig(f"Corner_{base_filename}.png", dpi=600, bbox_inches='tight', format='png')  # Higher DPI for better quality
        plt.show()
    
    
    if model_predictive:
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
        plt.vlines(x=times_map, ymin=-6000, ymax=24000,color='red', linestyle='dashed', linewidth=1.5)
        plt.savefig(f"Model Predictive {base_filename}.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
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
        print("MAP Params")
        print(map_params)
        map_predictive = []
        for sample in [map_params]:
            betas = sample[:n_betas]  # Estrai i beta
            times = sample[n_betas:n_betas+n_t_changes]  # Estrai i t_change
            if not fixed_sigma:
                initial_sigma=sample[n_betas+n_t_changes:]
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
    if posterior_predictive:        
        #Plot posterior predictiv
        n_posterior_samples = 200
        positions=np.random.choice(len(samples), n_posterior_samples, replace=False)
        posterior_samples = samples[positions]
        log_prob_samples=log_prob[positions]
        model_predictive = []
        map_params, map_log_like=get_maps(posterior_samples, log_prob_samples)
        times_map = map_params[n_betas:n_betas+n_t_changes]
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
        plt.xlabel("Time (days)", fontsize=16)
        plt.ylabel("Infected Individuals", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
        params = {'mathtext.default': 'regular'}          
        plt.rcParams.update(params)
        if n_betas!=1:
            plt.vlines(x=times_map, ymin=-1000, ymax=24000,color='red', linestyle='dashed', linewidth=1.5, label='MAP $t_{change}$')
        plt.legend(fontsize=16, loc='upper left')        
        plt.savefig(f"Posterior Predictive {base_filename}.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
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
        plt.legend()
        plt.grid()
        plt.show()
    
    if traces:
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
        params = {'mathtext.default': 'regular', 'xtick.labelsize': 14, 'ytick.labelsize': 14}          
        plt.rcParams.update(params)
        if n_betas==1:
            labels = [f"\u03B2" for i in range(n_betas)] + ["$t_{change}$" for i in range(n_betas-1)] + [f"\u03C3" for i in range(n_betas)]        
        else:
            labels = [f"\u03B2{to_subscript(i+1)}" for i in range(n_betas)] + ["$t_{change}$" for i in range(n_betas-1)] + [f"\u03C3{to_subscript(i+1)}" for i in range(n_betas)]        
        fig, axes = plt.subplots(n_plots, figsize=(15, 3*n_plots), sharex=True)
        for i in range(n_plots-1):
            ax = axes[i]
            ax.plot(samples_traces[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, samples_traces.shape[0])
            ax.set_ylabel(labels[i], fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=23, width=1.5, length=6)
            ax.grid()
            print(i)
            ax.set_xticklabels((100000+ax.get_xticks() * 10).astype(int))
        
        # Plot the log-likelihood trace
        axes[n_plots-1].plot(log_prob_3D_traces, "k", alpha=0.3)
        axes[n_plots-1].set_ylabel("Log-likelihood", fontsize=25)
        axes[n_plots-1].set_xlabel("Step number", fontsize=25)
        axes[n_plots-1].grid()
        print(n_plots-1)
        xticks = axes[n_plots-1].get_xticks()
        axes[n_plots-1].set_xticklabels((100000+xticks * 10).astype(int))
        plt.tight_layout()
        plt.tick_params(axis='both', which='major', labelsize=23, width=1.5, length=6)
        plt.savefig(f"Traces_{base_filename}.png", dpi=300, bbox_inches='tight', format='png')  # Higher DPI for better quality
        plt.show()
        
        
#%% 1 beta
# Get the samples
n_betas1=1
n_sigmas1=1
n_t_changes1=0

base_filename1='1beta_inferr_sigma'
filename1 = "1beta_inferr_sigma_euler.h5"
sampler1=emcee.backends.HDFBackend(filename1)
tau1 = sampler1.get_autocorr_time()
max_autocorr1 =max(tau1)
discard1=int(max_autocorr1*5)

thin=10

#Corner no R0
samples1 = sampler1.get_chain(discard=discard1, thin=thin, flat=True)
samples1_3D = sampler1.get_chain(discard=discard1, thin=thin, flat=False)

log_prob_1_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=True)
log_prob_1= log_prob_1_dirty[log_prob_1_dirty != -np.inf]

log_prob_1_3D_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=False)
slices_with_neg_inf1 = np.any(np.isinf(log_prob_1_3D_dirty), axis=0)
log_prob_1_3D = log_prob_1_3D_dirty[:, ~slices_with_neg_inf1]

#%% 2 betas
# Get the samples
n_betas2=2
n_sigmas2=2
n_t_changes2=1

base_filename2='2beta_inferr_sigma'
filename2 = "2beta_inferr_sigma_euler.h5"
sampler2=emcee.backends.HDFBackend(filename2)
tau2 = sampler2.get_autocorr_time()
max_autocorr2 =max(tau2)
discard2=int(max_autocorr2*5)

thin=10

#Corner no R0
samples2 = sampler2.get_chain(discard=discard2, thin=thin, flat=True)
samples2_3D = sampler2.get_chain(discard=discard2, thin=thin, flat=False)

log_prob_2_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=True)
log_prob_2 = log_prob_2_dirty[log_prob_2_dirty != -np.inf]

log_prob_2_3D_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=False)
slices_with_neg_inf2 = np.any(np.isinf(log_prob_2_3D_dirty), axis=0)
log_prob_2_3D = log_prob_2_3D_dirty[:, ~slices_with_neg_inf2]




#%% 3 betas
# Get the samples
n_betas3=3
n_sigmas3=3
n_t_changes3=2

base_filename3='3beta_inferr_sigma'
filename3 = "3beta_inferr_sigma_euler.h5"
sampler3=emcee.backends.HDFBackend(filename3)
#tau3 = sampler3.get_autocorr_time()
max_autocorr3 =30000#max(tau3)
discard3=int(max_autocorr3*5)

thin=10

#Corner no R0
samples3 = sampler3.get_chain(discard=discard3, thin=thin, flat=True)
samples3_3D = sampler3.get_chain(discard=discard3, thin=thin, flat=False)

log_prob_3_dirty=sampler3.get_log_prob(discard=discard3, thin=thin, flat=True)
log_prob_3 = log_prob_3_dirty[log_prob_3_dirty != -np.inf]

log_prob_3_3D_dirty=sampler3.get_log_prob(discard=discard3, thin=thin, flat=False)
slices_with_neg_inf3= np.any(np.isinf(log_prob_3_3D_dirty), axis=0)
log_prob_3_3D = log_prob_3_3D_dirty[:, ~slices_with_neg_inf1]

#%% 1 beta_1
# Get the samples
n_betas1=1
n_sigmas1=1
n_t_changes1=0

base_filename1='1beta_inferr_sigma'
filename1 = "1beta_inferr_sigma_1_euler.h5"
sampler1=emcee.backends.HDFBackend(filename1)
tau1 = sampler1.get_autocorr_time()
max_autocorr1 =max(tau1)
discard1=int(max_autocorr1*5)

thin=10

samples_1_corner=sampler1.get_chain(discard=100000, thin=thin, flat=True)
samples_1_traces=sampler1.get_chain(discard=100000, thin=thin, flat=False)

#Corner no R0
samples1 = sampler1.get_chain(discard=discard1, thin=thin, flat=True)
samples1_3D = sampler1.get_chain(discard=discard1, thin=thin, flat=False)

log_prob_1_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=True)
log_prob_1= log_prob_1_dirty[log_prob_1_dirty != -np.inf]

log_prob_1_3D_dirty=sampler1.get_log_prob(discard=discard1, thin=thin, flat=False)
slices_with_neg_inf1 = np.any(np.isinf(log_prob_1_3D_dirty), axis=0)
log_prob_1_3D = log_prob_1_3D_dirty[:, ~slices_with_neg_inf1]

log_prob_1_3D_traces_dirty=sampler1.get_log_prob(discard=100000, thin=thin, flat=False)
slices_with_neg_inf1 = np.any(np.isinf(log_prob_1_3D_traces_dirty), axis=0)
log_prob_1_3D_traces = log_prob_1_3D_traces_dirty[:, ~slices_with_neg_inf1]


analysis_indices1 = np.random.choice(np.arange(0, samples1_3D.shape[0], dtype=int), size=500, replace=False)
subsample1_3D_analysis=samples1_3D[analysis_indices1,:, :]
subsample1_analysis=subsample1_3D_analysis.reshape(25000, 2)

#%% 2 betas_1
# Get the samples
n_betas2=2
n_sigmas2=2
n_t_changes2=1

base_filename2='2beta_inferr_sigma'
filename2 = "2beta_inferr_sigma_1_euler.h5"
sampler2=emcee.backends.HDFBackend(filename2)
tau2 = sampler2.get_autocorr_time()
max_autocorr2 =max(tau2)
discard2=int(max_autocorr2*5)

thin=10

#Corner no R0
samples2 = sampler2.get_chain(discard=discard2, thin=thin, flat=True)
samples2_3D = sampler2.get_chain(discard=discard2, thin=thin, flat=False)

samples_2_corner=sampler2.get_chain(discard=100000, thin=thin, flat=True)
samples_2_traces=sampler2.get_chain(discard=100000, thin=thin, flat=False)

log_prob_2_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=True)
log_prob_2 = log_prob_2_dirty[log_prob_2_dirty != -np.inf]

log_prob_2_3D_dirty=sampler2.get_log_prob(discard=discard2, thin=thin, flat=False)
slices_with_neg_inf2 = np.any(np.isinf(log_prob_2_3D_dirty), axis=0)
log_prob_2_3D = log_prob_2_3D_dirty[:, ~slices_with_neg_inf2]

log_prob_2_3D_traces_dirty=sampler2.get_log_prob(discard=100000, thin=thin, flat=False)
slices_with_neg_inf2 = np.any(np.isinf(log_prob_2_3D_traces_dirty), axis=0)
log_prob_2_3D_traces = log_prob_2_3D_traces_dirty[:, ~slices_with_neg_inf2]


analysis_indices2 = np.random.choice(np.arange(0, samples2_3D.shape[0], dtype=int), size=500, replace=False)
subsample2_3D_analysis=samples2_3D[analysis_indices2,:, :]
subsample2_analysis=subsample2_3D_analysis.reshape(25000, 5)

#%% 3 betas_1
# Get the samples
n_betas3=3
n_sigmas3=3
n_t_changes3=2

base_filename3='3beta_inferr_sigma'
filename3 = "3beta_inferr_sigma_1_euler.h5"
sampler3=emcee.backends.HDFBackend(filename3)
#tau3 = sampler3.get_autocorr_time()
max_autocorr3 =15189#max(tau3)
discard3=int(max_autocorr3*5)

thin=10

#Corner no R0
samples3_3D_raw = sampler3.get_chain(discard=discard3, thin=thin, flat=False)
mask_3D=samples3_3D_raw[:, :, 5] <1000
mask_3D_1D = np.all(mask_3D, axis=0)
samples3_3D = samples3_3D_raw[:,mask_3D_1D, :]

samples3_raw = sampler3.get_chain(discard=discard3, thin=thin, flat=True)
mask_2D=samples3_raw[:, 5] <1000
samples3=samples3_raw[mask_2D, :]

samples_3_corner_raw=sampler3.get_chain(discard=300000, thin=3, flat=True)
mask_2D_corner=samples_3_corner_raw[:, 5] <1000
samples_3_corner=samples_3_corner_raw[mask_2D_corner, :]

samples_3_traces_raw=sampler3.get_chain(discard=0, thin=thin, flat=False)
mask_3D_traces=samples_3_traces_raw[:, :, 5] <1000
mask_3D_1D_traces = np.all(mask_3D_traces, axis=0)
samples3_traces = samples_3_traces_raw[:,mask_3D_1D_traces, :]

log_prob_3_dirty_raw=sampler3.get_log_prob(discard=discard3, thin=thin, flat=True)
log_prob_3_dirty=log_prob_3_dirty_raw[mask_2D]
log_prob_3 = log_prob_3_dirty[log_prob_3_dirty != -np.inf]

log_prob_3_3D_dirty_raw=sampler3.get_log_prob(discard=discard3, thin=thin, flat=False)
log_prob_3_3D_dirty=log_prob_3_3D_dirty_raw[:, mask_3D_1D]
slices_with_neg_inf3= np.any(np.isinf(log_prob_3_3D_dirty), axis=0)
log_prob_3_3D = log_prob_3_3D_dirty[:, ~slices_with_neg_inf3]

log_prob_3_3D_traces_dirty_raw=sampler3.get_log_prob(discard=0, thin=thin, flat=False)
log_prob_3_3D_traces_dirty=log_prob_3_3D_traces_dirty_raw[:, mask_3D_1D]
slices_with_neg_traces_inf3= np.any(np.isinf(log_prob_3_3D_traces_dirty), axis=0)
log_prob_3_3D_traces = log_prob_3_3D_traces_dirty[:, ~slices_with_neg_traces_inf3]



analysis_indices3 = np.random.choice(np.arange(0, samples3_3D.shape[0], dtype=int), size=500, replace=False)
subsample3_3D_analysis=samples3_3D[analysis_indices3,:, :]
subsample3_analysis=subsample3_3D_analysis.reshape(subsample3_3D_analysis.shape[1]*500, 8)
#%%


plot_del_diocane(samples1,samples_1_corner, samples_1_traces, samples1_3D, log_prob_1,log_prob_1_3D, 
                 log_prob_1_3D_traces, n_betas1, n_sigmas1, n_t_changes1, base_filename1, 
                 fixed_sigma=False, initial_sigma=None, corner_plot=False, posterior_predictive=False, traces=True)

plot_del_diocane(samples2, samples_2_corner, samples_2_traces,samples2_3D, log_prob_2,log_prob_2_3D,
                 log_prob_2_3D_traces, n_betas2, n_sigmas2, n_t_changes2, base_filename2, 
                 fixed_sigma=False, initial_sigma=None, corner_plot=False, posterior_predictive=False, traces=True)

plot_del_diocane(samples3,samples_3_corner, samples3_traces, samples3_3D, log_prob_3,log_prob_3_3D,log_prob_3_3D_traces, n_betas3, n_sigmas3, n_t_changes3, base_filename3, fixed_sigma=False, initial_sigma=None, corner_plot=True)

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


def get_ppd(fixed_sigma, n_t_changes, n_betas, subsamples, initial_sigma=None):


    model_predictive=[]
    posterior_predictive = []
    sigma_single=[]
    for sample in subsamples:
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

ppd1, model_predictive1, sigmas1 = get_ppd(False, n_t_changes1, n_betas1, subsample1_analysis)
pte1, t_rep1, t_data1 = ppd_model_check(test_statistic, sigmas1, cases, ppd1, model_predictive1)
print(pte1)

ppd2, model_predictive2, sigmas2 = get_ppd(False, n_t_changes2, n_betas2, subsample2_analysis)
pte2, t_rep2, t_data2 = ppd_model_check(test_statistic, sigmas2, cases, ppd2, model_predictive2)
print(pte2)

ppd3, model_predictive3, sigmas3 = get_ppd(False, n_t_changes3, n_betas3, subsample3_analysis)
pte3, t_rep3, t_data3 = ppd_model_check(test_statistic, sigmas3, cases, ppd3, model_predictive3)
print(pte3)
#Percentage of posterior samples where the posterior is further away from the model than the real cases.
#%%#WAIC, DIC and bayes ratio jaco
def lppd_waic(theta, t, N, case, n_betas, n_t_changes):
    lppd=np.zeros(len(cases))
    betas = theta[:n_betas]  # Estrai i beta
    t_changes = theta[n_betas:n_betas+n_t_changes]  # Estrai i t_change
    sigma=theta[n_betas+n_t_changes:]
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
    if n_t_changes>0:
        sigmas_t = np.concatenate([np.full(int(t_changes[i]) - int(t_changes[i-1]) if i > 0 else int(t_changes[0]), sigma[i]) 
                                   for i in range(len(sigma) - 1)] + [np.full(len(cases) - int(t_changes[-1]), sigma[-1])])
    else:
        sigmas_t = np.concatenate([np.full(int(t_changes[i]) - int(t_changes[i-1]) if i > 0 else int(t_changes[0]), sigma[i]) 
                                   for i in range(len(sigma) - 1)] + [np.full(len(cases), sigma[-1])])
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
    return lppd

def waic(subsamples_3D, n_betas, n_t_changes, random_numbers):
    blobs = np.empty((500, subsamples_3D.shape[1], 30))
    for i, n in enumerate(random_numbers):
        for j in range(0,subsamples_3D.shape[1]):
            theta=subsamples_3D[i, j, :]
            lppd_theta=lppd_waic(theta, t, population, cases, n_betas, n_t_changes)
            blobs[i, j, :]=lppd_theta
    slices_with_neg_inf = np.any(np.isinf(blobs), axis=(0, 2))
    timewise_log_likelihood = blobs[:, ~slices_with_neg_inf, :]
    lppd_chain = np.log(np.mean(np.exp(timewise_log_likelihood), axis=0))
    p_waic_chain = np.var(timewise_log_likelihood, axis=0)
    # Check if the any of the terms in p_waic are too large, which indicates
    # a problem
# =============================================================================
#     if np.any(p_waic_chain > 0.4):
#         print(f"Warning: Var[log p(y_i|theta)] > 0.4 for data points "
#               f"{np.argwhere(p_waic_chain > 0.4)}. p_WAIC unreliable!")
# =============================================================================
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


waic1=waic(subsample1_3D_analysis, n_betas1, n_t_changes1, analysis_indices1)

waic2=waic(subsample2_3D_analysis, n_betas2, n_t_changes2, analysis_indices2)

waic3=waic(subsample3_3D_analysis, n_betas3, n_t_changes3, analysis_indices3)


print('waic1:',waic1,'waic2:', waic2, 'waic1/waic2:',waic1/waic2)



log_prob_1_top, log_prob_2_top=bayes_ratio(samples1, log_prob_1, samples2, log_prob_2)

log_prob_2_top, log_prob_3_top=bayes_ratio(samples2, log_prob_2, samples3, log_prob_3)

print('Log_prob§',log_prob_1_top,'Log_prob2:', log_prob_2_top, 'Bayes_ratio1/2:',log_prob_1_top/log_prob_2_top)
print('Log_prob2',log_prob_1_top,'Log_prob3:', log_prob_3_top, 'Bayes_ratio1/3:',log_prob_1_top/log_prob_3_top)
print('Log_prob2',log_prob_2_top,'Log_prob3:', log_prob_3_top, 'Bayes_ratio2/3:',log_prob_2_top/log_prob_3_top)


DIC_1, p_V_1, p_D_1=DIC_comparison(samples1, log_prob_1)
DIC_2, p_V_2, p_D_2=DIC_comparison(samples2, log_prob_2)
DIC_3, p_V_3, p_D_3=DIC_comparison(samples3, log_prob_3)


print('DIC1:',DIC_1, 'p_V_1',p_V_1 ,'p_D_1',p_D_1 ,'DIC2:', DIC_2,'p_V_1:', p_V_2,'p_D_2:', p_D_2 ,'DIC1/DIC2:',DIC_1/DIC_2)
print('DIC1:',DIC_1, 'p_V_1',p_V_1 ,'p_D_1',p_D_1 ,'DIC1:', DIC_1,'p_V_3:', p_V_3,'p_D_3:', p_D_3 ,'DIC1/DIC3:',DIC_1/DIC_3)
print('DIC2:',DIC_2, 'p_V_2',p_V_2 ,'p_D_2',p_D_2 ,'DIC3:', DIC_3,'p_V_3:', p_V_3,'p_D_3:', p_D_3 ,'DIC2/DIC3:',DIC_2/DIC_3)

#%%R0
#Model 1
samples_1_R0=samples_1_traces.reshape(40000 * 50, 2)
R0_1=samples_1_R0[:, 0]*14
R0_1_mean=np.mean(R0_1)
R0_1_lower_bound = np.percentile(R0_1, 2.5)
R0_1_upper_bound = np.percentile(R0_1, 97.5)
R0_1_diff_low=R0_1_mean-R0_1_lower_bound
R0_1_diff_high=R0_1_upper_bound-R0_1_mean

print(R0_1_mean, R0_1_diff_low, R0_1_diff_high)

#Model 2
samples_2_R0=samples_2_traces.reshape(40000 * 50, 5)
R0_2_1=samples_2_R0[:, 0]*14
R0_2_1_mean=np.mean(R0_2_1)
R0_2_1_lower_bound = np.percentile(R0_2_1, 2.5)
R0_2_1_upper_bound = np.percentile(R0_2_1, 97.5)
R0_2_1_diff_low=R0_2_1_mean-R0_2_1_lower_bound
R0_2_1_diff_high=R0_2_1_upper_bound-R0_2_1_mean

print(R0_2_1_mean, R0_2_1_diff_low, R0_2_1_diff_high)
print(R0_2_1_mean, R0_2_1_lower_bound, R0_2_1_upper_bound)

R0_2_2=samples_2_R0[:, 1]*14
R0_2_2_mean=np.mean(R0_2_2)
R0_2_2_lower_bound = np.percentile(R0_2_2, 2.5)
R0_2_2_upper_bound = np.percentile(R0_2_2, 97.5)
R0_2_2_diff_low=R0_2_2_mean-R0_2_2_lower_bound
R0_2_2_diff_high=R0_2_2_upper_bound-R0_2_2_mean

print(R0_2_2_mean, R0_2_2_diff_low, R0_2_2_diff_high)
print(R0_2_2_mean, R0_2_2_lower_bound, R0_2_2_upper_bound)

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



