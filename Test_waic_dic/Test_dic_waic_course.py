# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:48:07 2024

@author: Ninja
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import emcee

import jax
jax.config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
from jax.lax import cond
from jax import jit, hessian, vmap

import jaxopt

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import getdist
import getdist.plots

from analyse import plot_data, analyse_data

import warnings
warnings.filterwarnings('ignore')
from IPython import get_ipython  # For IPython utilities

get_ipython().run_line_magic('matplotlib', 'inline')

#%% traces
n_param = 3
param_names = ["Omega_m", "H0", "M"]

# Here we read a file with a chain that was run before
chain_per_walker = np.load("emcee_chain.npy")
assert chain_per_walker.shape == (5000, 9, 3)

n_walker = 9
param_names = ["Omega_m", "H0", "M"]

# Plot the tracer of the chain, this time for all the walkers
fig, ax = plt.subplots(len(param_names), 1, sharex=True)
plt.subplots_adjust(hspace=0)

for i, name in enumerate(param_names):
    for j in range(n_walker):
        ax[i].plot(chain_per_walker[:, j, i], lw=0.5, alpha=0.7, color=f"C{j}")
    ax[i].set_ylabel(name)

ax[-1].set_xlabel("step");

#%% autocorrelation
# If the chain is an array (for example from a different MCMC algorithm)
tau = emcee.autocorr.integrated_time(chain_per_walker)

# If the algorithm does use walkers, set the has_walkers=False argument
# tau = emcee.autocorr.integrated_time(chain_per_walker[:, 0], has_walkers=False)

# Or directly from emcee:
# tau = sampler.get_autocorr_time()

print("Integrated auto-correlation time")
for name, iat in zip(param_names, tau):
    print(f"{name}: {iat:.1f}")
    
max_autocorr = max(tau)
burn_in = int(5*max_autocorr)
thin = int(max_autocorr/2)

chain = chain_per_walker[burn_in::thin].reshape(-1, n_param)
# Directly from an emcee sampler:
# chain = sampler.get_chain(discard=burn_in, thin=thin, flat=True)


#%%Interpreting posteriors
distr = scipy.stats.gamma(a=2, scale=1/4)

alpha = 1-0.68

x = np.linspace(0, 2, 500)

ci = distr.ppf(alpha/2), distr.ppf(1 - alpha/2)

plt.plot(x, distr.pdf(x))
x_ci = np.linspace(ci[0], ci[1], 100)
plt.fill_between(x_ci, 0, distr.pdf(x_ci), alpha=0.5)

plt.axvline(x=ci[0], c="grey", label="Equal-tail interval")
plt.axvline(x=ci[1], c="grey")

plt.ylim(bottom=0)

plt.xlabel("$x$")
plt.ylabel("$P(x)$")

_ = plt.legend()


def find_ci(alpha, pdf_fn, mode, support):
    def c_to_alpha(c):
        ci_L = scipy.optimize.root_scalar(
            f=lambda x: c-pdf_fn(x),
            bracket=(support[0], mode)).root
        ci_R = scipy.optimize.root_scalar(
            f=lambda x: c-pdf_fn(x),
            bracket=(mode, support[1])).root
        alpha = distr.cdf(ci_R) - distr.cdf(ci_L)
        return alpha, (ci_L, ci_R)
    
    min_c = max(pdf_fn(support[0]), pdf_fn(support[1]))

    c = scipy.optimize.root_scalar(
        f=lambda c: c_to_alpha(c)[0] - alpha,
        bracket=(min_c, pdf_fn(mode))).root
    
    return c_to_alpha(c)[1]

mode = scipy.optimize.minimize(lambda x: -distr.logpdf(x), x0=1.0).x.squeeze()

ci = find_ci(alpha=0.68, pdf_fn=distr.pdf, mode=mode, support=(0, 2))

plt.plot(x, distr.pdf(x))
x_ci = np.linspace(ci[0], ci[1], 100)
plt.fill_between(x_ci, 0, distr.pdf(x_ci), alpha=0.5)

plt.axvline(x=ci[0], c="grey", label="Highest-density interval")
plt.axvline(x=ci[1], c="grey")

plt.ylim(bottom=0)

plt.xlabel("$x$")
plt.ylabel("$P(x)$")

_ = plt.legend()



def model(x, a, b, c, d):
    x0=0.5
    return a + b * x * ((1+x)/(1+x0))**c * ((1+x**2)/(1+x0**2))**d


a_prior = tfd.Uniform(low=-10, high=10)
b_prior = tfd.Uniform(low=-10, high=10)
c_prior = tfd.Uniform(low=-1000, high=1000)
d_prior = tfd.Uniform(low=-1000, high=1000)


# Generate synthetic data
np.random.seed(42)
x = jnp.linspace(0, 1, 10)
true_params = [2, 0.1, 0, 0]
y = model(x, *true_params)
y_err = 0.05 * jnp.ones_like(y)


def create_likelihood_distribution(model, y_err):
    def distr(params, x):
        mu = model(x, *params)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=y_err)

    return distr


def log_likelihood(params):
    return create_likelihood_distribution(model=model, y_err=y_err)(params, x).log_prob(y)

def log_prior(params):
    return a_prior.log_prob(params[0]) + b_prior.log_prob(params[1]) + c_prior.log_prob(params[2]) + d_prior.log_prob(params[3])

# Define the log posterior
def log_posterior(params):
    return log_likelihood(params) + log_prior(params)

params_initial = jnp.array([1.0, 1.0, 1.0, 1.0])
param_names = ["a", "b", "c", "d"]

solver = jaxopt.ScipyMinimize(fun=jit(lambda x: -log_posterior(x)), method="L-BFGS-B")
solution = solver.run(params_initial)
MAP_params = solution.params
cov = jnp.linalg.inv(-hessian(log_posterior)(MAP_params))

print("Model 1")
for name, p, p_std, p_true in zip(param_names, MAP_params, np.sqrt(np.diag(cov)), true_params):
    print(f"{name} = {p:.2f} ± {p_std:.2f} (truth: {p_true:.2f})")
    

# Set up the initial parameters for the MCMC
initial_params = np.array([1.0, 1.0, 1.0, 1.0])
ndim = len(initial_params)
nwalkers = 32
nsteps = 10000

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, jit(vmap(log_posterior)), vectorize=True)

# Initialize the walkers
p0 = np.random.normal(initial_params, scale=0.1, size=(nwalkers, ndim))

# Run the MCMC
sampler.run_mcmc(p0, nsteps, progress=True)

max_autocorr = max(sampler.get_autocorr_time())
# Extract the samples
samples = sampler.get_chain(discard=int(5*max_autocorr), thin=int(max_autocorr/2), flat=True)
getdist_samples = getdist.MCSamples(
    samples=samples,
    names=param_names,
    label="Model 1",
    ranges={"c": (-1000, 1000), "d": (-1000, 1000)}
)

g = getdist.plots.get_single_plotter()

g.triangle_plot([getdist_samples],
                params=["a", "b"],
                filled=True,
                markers={k: v for k, v in zip(param_names, MAP_params)})

g.subplots[0,0].plot([],[], **{"lw" : 1.0, "ls" : "--", "c" : "k", "alpha" : 0.5}, label="Posterior mode")
g.subplots[0,0].legend(frameon=False, loc=2, bbox_to_anchor=(1,1));

g = getdist.plots.get_single_plotter()

fig = g.triangle_plot([getdist_samples], filled=True,
                 markers={k: v for k, v in zip(param_names, MAP_params)})
g.subplots[0,0].plot([],[], **{"lw" : 1.0, "ls" : "--", "c" : "k", "alpha" : 0.5}, label="Posterior mode")
g.subplots[0,0].legend(frameon=False, loc=2, bbox_to_anchor=(1,1));

def log_p(p):
    x, y = p
    if y < 0 or x < 0 or y > x or y < x/2:
        return -np.inf
    
    return -(x+y)**2

theta_fid = (0.5, 0.5)
nwalkers = 32
ndim = len(theta_fid)

n_sample = 12000
burn_in = 2000

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_p)

sampler.run_mcmc(np.random.normal(loc=0, scale=0.1, size=(nwalkers,ndim))+theta_fid, n_sample, progress=True)
max_autocorr = max(sampler.get_autocorr_time())

samples = getdist.MCSamples(name_tag="Full",
                            samples=sampler.get_chain(flat=True, discard=int(5*max_autocorr), thin=int(max_autocorr/2)),
                            names=["theta0", "theta1"], labels=["x", "y"],
                            ranges={"theta0" : [0,4], "theta1" : [0,4],}
                                   )



def log_p(p):
    x, y = p
    if y < 0 or x < 0 or y > x or y < x/2:
        return -np.inf
    
    return -(x+y)**2

theta_fid = (0.5, 0.5)
nwalkers = 32
ndim = len(theta_fid)

n_sample = 12000
burn_in = 2000

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_p)

sampler.run_mcmc(np.random.normal(loc=0, scale=0.1, size=(nwalkers,ndim))+theta_fid, n_sample, progress=True)
max_autocorr = max(sampler.get_autocorr_time())

samples = getdist.MCSamples(name_tag="Full",
                            samples=sampler.get_chain(flat=True, discard=int(5*max_autocorr), thin=int(max_autocorr/2)),
                            names=["theta0", "theta1"], labels=["x", "y"],
                            ranges={"theta0" : [0,4], "theta1" : [0,4],}
                                   )

MAP = [0,0]

samples.smooth_scale_1D = 0.2
samples.smooth_scale_2D = 0.2

g = getdist.plots.get_single_plotter()
g.settings.legend_frame = False


g.triangle_plot([samples],
                filled_compare=True,
                params=["theta0", "theta1"],
                markers={"theta0" : MAP[0], 
                         "theta1" : MAP[1],
                         },
               )

g.subplots[0,0].set_xlim(left=-0.5)
g.subplots[1,1].set_xlim(left=-0.5)

g.subplots[0,0].plot([],[], **{"lw" : 1.0, "ls" : ":", "c" : "k", "alpha" : 0.5}, label="Mode of $p(x, y)$")
g.subplots[0,0].legend(frameon=False, loc=2, bbox_to_anchor=(1,1));


#%%Model checking
x, y, y_err = np.loadtxt("data_2.txt", unpack=True)
a_true, b_true, c_true,  = -0.15, 0.7, -0.45

plot_data(x, y, y_err);

def make_gaussian_likelihood_components(model, log_prior):
    """Create the log_likelihood, log_posterior, predict functions from a
    model and log_prior"""
    def log_likelihood(y, theta, x, sigma_y):
        prediction = model(theta, x)

        return (
            -0.5 * np.sum((y - prediction)**2/sigma_y**2)    # Exponent
            -0.5 * np.sum(np.log(2*np.pi*sigma_y**2))        # Normalisation
        )
    
    def log_posterior(theta, x, sigma_y, y):
        return log_likelihood(y, theta, x, sigma_y) + log_prior(theta)
    
    def predict(theta, x, sigma_y):
        mu = model(theta, x)
        return np.random.normal(loc=mu, scale=sigma_y)
    
    return log_likelihood, log_posterior, predict
def linear_model(theta, x):
    m, b = theta
    return m*x + b

def log_prior_linear_model(theta):
    m, b = theta
    # Unnormalised uniform prior m\simU(-2, 2), b \sim U(-3, 3)
    if -2 < m < 2 and -3 < b < 3:
        return 0
    else:
        return -np.inf

log_likelihood_lin, log_posterior_lin, predict_lin = \
    make_gaussian_likelihood_components(
        linear_model, log_prior_linear_model
    )
    
# From bayesian_stats_course_tools.analyse
# This function returns a dict with the following entries:
# - MAP: parameter values at the MAP
# - PPD: samples from the posterior predictive distribution->predict_lin used to generate data with random parameter samples from the posterior-> same as linear_model but with noise sigma
# - PPD_params: the parameter samples for the samples in the PPD and TPD
# - TPD: samples from the TPD (samples of model predictions) ->linear_model used to generate data with random parameter samples from the posterior
# - chain: MCMC chain of the parameters
results_linear = analyse_data(
    data=dict(x=x, y=y, y_err=y_err),
    log_posterior_fn=log_posterior_lin,
    model_fn=linear_model,
    predict_fn=predict_lin,
    param_names=["m", "b"],
    theta_init=[1, 0],
    plot=False
)

def quadratic_model(theta, x):
    a, b, c = theta
    return a*x**2 + b*x + c

def log_prior_quadratic_model(theta):
    a, b, c = theta
    # Unnormalised uniform prior m\simU(-2, 2), b \sim U(-3, 3)
    if -1 < a < 1 and -2 < b < 2 and -3 < c < 3:
        return 0
    else:
        return -np.inf

log_likelihood_quad, log_posterior_quad, predict_quad = \
    make_gaussian_likelihood_components(
        quadratic_model, log_prior_quadratic_model
    )

results_quadratic = analyse_data(
    data=dict(x=x, y=y, y_err=y_err),
    log_posterior_fn=log_posterior_quad,
    model_fn=quadratic_model,
    predict_fn=predict_quad,
    param_names=["a", "m", "b"],
    theta_init=[0.1, 1, 0],
    plot=False
)
plot_data(
    x, y, y_err,
    models=[
        dict(x=x, y=linear_model(results_linear["MAP"], x), style=dict(label="MAP linear model")),
        dict(x=x, y=quadratic_model(results_quadratic["MAP"], x), style=dict(label="MAP quadratic model"))
    ]
);

#%%Chi-squared goodness of fit-ONly consider MAP
def chi_squared(y, sigma_y, mu):
    return np.sum((y - mu)**2/sigma_y**2)

chi_squared_linear = chi_squared(
    y, y_err,
    mu=linear_model(results_linear["MAP"], x)
)
chi_squared_quadratic = chi_squared(
    y, y_err,
    mu=quadratic_model(results_quadratic["MAP"], x)
)
n_data = len(y)
n_param_lin = 2
n_param_quad = 3

PTE_lin = scipy.stats.chi2(df=n_data - n_param_lin).sf(chi_squared_linear)
PTE_quad = scipy.stats.chi2(df=n_data - n_param_quad).sf(chi_squared_quadratic)

print(f"Linear: χ²={chi_squared_linear:.1f}, "
      f"ndof={n_data}-{n_param_lin}, PTE={PTE_lin:.3f}")

print(f"Quadratic: χ²={chi_squared_quadratic:.1f}, "
      f"ndof={n_data}-{n_param_quad}, PTE={PTE_quad:.3f}")

n_data = len(y)
n_param_lin = 2
n_param_quad = 3

PTE_lin = scipy.stats.chi2(df=n_data - n_param_lin).sf(chi_squared_linear)
PTE_quad = scipy.stats.chi2(df=n_data - n_param_quad).sf(chi_squared_quadratic)

print(f"Linear: χ²={chi_squared_linear:.1f}, "
      f"ndof={n_data}-{n_param_lin}, PTE={PTE_lin:.3f}")

print(f"Quadratic: χ²={chi_squared_quadratic:.1f}, "
      f"ndof={n_data}-{n_param_quad}, PTE={PTE_quad:.3f}")

#%%posterior predictive check
def test_statistic(y, theta, x, sigma_y, model): #in this case chi squared but can be anything else
    mu = model(theta, x)
    t = chi_squared(y, sigma_y, mu)
    return t

def ppd_model_check(test_statistic, y, ppd, ppd_params):
    t_data = []
    t_rep = []
    for y_rep, theta in zip(ppd, ppd_params):
        t_data.append(test_statistic(y, theta))
        t_rep.append(test_statistic(y_rep, theta))

    t_data = np.array(t_data)
    t_rep = np.array(t_rep)

    pte = (t_rep >= t_data).sum()/len(t_data)
    return pte, t_rep, t_data
PPD_PTE_lin, t_rep_lin, t_data_lin = ppd_model_check(
    test_statistic=lambda y, theta: test_statistic(
        y, theta, x, y_err, linear_model),
    y=y,
    ppd=results_linear["PPD"],
    ppd_params=results_linear["PPD_params"]
)

PPD_PTE_quad, t_rep_quad, t_data_quad = ppd_model_check(
    test_statistic=lambda y, theta: test_statistic(
        y, theta, x, y_err, quadratic_model),
    y=y,
    ppd=results_quadratic["PPD"],
    ppd_params=results_quadratic["PPD_params"]
)

print(f"Linear: PPD PTE={PPD_PTE_lin:.3f}")
print(f"Quadratic: PPD PTE={PPD_PTE_quad:.3f}")

#%%Model comparison

def DIC(theta_star, theta_samples, log_likelihood):
    # Compute log likelihood at theta_star and the samples theta_i
    log_likelihood_star = log_likelihood(theta_star)
    log_likelihood_samples = np.array(
        [log_likelihood(theta) for theta in theta_samples]
    )
    p_D = 2*(log_likelihood_star - np.mean(log_likelihood_samples))
    p_V = 2*np.var(log_likelihood_samples)
    return -2*(log_likelihood_star - p_D), p_D, p_V

DIC_lin, p_D_lin, p_V_lin = DIC(
    theta_star=results_linear["MAP"],
    theta_samples=results_linear["PPD_params"],
    log_likelihood=lambda theta: log_likelihood_lin(y, theta, x, y_err)
)

DIC_quad, p_D_quad, p_V_quad = DIC(
    theta_star=results_quadratic["MAP"],
    theta_samples=results_quadratic["PPD_params"],
    log_likelihood=lambda theta: log_likelihood_quad(y, theta, x, y_err)
)
print(f"Linear: DIC = {DIC_lin:.1f}, p_D = {p_D_lin:.1f}, p_V = {p_V_lin:.1f}")
print(f"Quadratic: DIC = {DIC_quad:.1f}, p_D = {p_D_quad:.1f}, p_V = {p_V_quad:.1f}")


def WAIC(theta_samples, log_likelihood, y_partitions, x_partitions, y_err_partitons):
    # Compute the log likelihood for each partition separately
    pointwise_log_likelihood_samples = np.array(
        [[log_likelihood(y_partitions[i], theta, x_partitions[i], y_err_partitons[i])
            for i in range(len(y_partitions))] 
         for theta in theta_samples]
    )

    # Compute the lppd and p_waic for each partition
    lppd = np.log(np.mean(np.exp(pointwise_log_likelihood_samples), axis=0))
    p_waic = np.var(pointwise_log_likelihood_samples, axis=0)
    # Check if the any of the terms in p_waic are too large, which indicates
    # a problem
    if np.any(p_waic > 0.4):
        print(f"Warning: Var[log p(y_i|theta)] > 0.4 for data points "
              f"{np.argwhere(p_waic > 0.4)}. p_WAIC unreliable!")
    # Sum up the partitions
    lppd = np.sum(lppd)
    p_waic = np.sum(p_waic)

    return -2*(lppd - p_waic), p_waic, pointwise_log_likelihood_samples

# =============================================================================
# pointwise_log_likelihood_samples = np.zeros((len(results_linear["PPD_params"]), len(y)))
# 
# for i in range(len(y)): 
#     for j, theta in enumerate(results_linear["PPD_params"]):
#         pointwise_log_likelihood_samples[j, i] = np.array(
#             log_likelihood_lin(y[i], theta, x[i], y_err[i]))
#   
# pointwise_log_likelihood_samples_1 = np.zeros((1, len(y)))
# 
# for i in range(len(y)): 
#     j=0
#     theta=results_linear["PPD_params"][-1,:]
#     pointwise_log_likelihood_samples_1[j, i] = np.array(
#         log_likelihood_lin(y[i], theta, x[i], y_err[i]))
#               
# log_likelihood_lin_1=log_likelihood_lin(y, results_linear["PPD_params"][-1,:], x, y_err)
# log_likelihood_lin_1_test=np.sum(pointwise_log_likelihood_samples_1)
# 
# =============================================================================
WAIC_lin, p_WAIC_lin, pointwise_log_likelihood_samples_lin = WAIC(
    theta_samples=results_linear["PPD_params"],
    log_likelihood=log_likelihood_lin,
    y_partitions=y,
    x_partitions=x,
    y_err_partitons=y_err
)

WAIC_quad, p_WAIC_quad, pointwise_log_likelihood_samples_quad = WAIC(
    theta_samples=results_quadratic["PPD_params"],
    log_likelihood=log_likelihood_quad,
    y_partitions=y,
    x_partitions=x,
    y_err_partitons=y_err
)
print(f"Linear: WAIC = {WAIC_lin:.1f}, p_WAIC = {p_WAIC_lin:.1f}")
print(f"Quadratic: WAIC = {WAIC_quad:.1f}, p_WAIC = {p_WAIC_quad:.1f}")