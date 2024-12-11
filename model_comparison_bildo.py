import numpy as np
import emcee



def get_maps(path_1, path_2):

    discard_1 = 1
    discard_2 = 1

    sampler_1 = emcee.backends.HDFBackend(path_1)
    log_prob_1 = sampler_1.get_log_prob(discard=discard_1, thin=1, flat=True)
    chain_1 = sampler_1.get_chain(discard=discard_1, thin=1, flat=True)  #shape (N, nparams)

    sampler_2 = emcee.backends.HDFBackend(path_2)
    log_prob_2 = sampler_2.get_log_prob(discard=discard_2, thin=1, flat=True)
    chain_2 = sampler_2.get_chain(discard=discard_2, thin=1, flat=True)  #shape (N, nparams)

    idx_map_1 = np.argmax(log_prob_1)
    theta_star_1 = chain_1[idx_map_1]

    idx_map_2 = np.argmax(log_prob_2)
    theta_star_2 = chain_2[idx_map_2]

    return theta_star_1, theta_star_2, log_prob_1, log_prob_2



def bayes_ratio(path_1, path_2):

    _, _, log_prob_1, log_prob_2 = get_maps(path_1, path_2)

    log_py_theta_star_1 = np.max(log_prob_1)
    log_py_theta_star_2 = np.max(log_prob_2)

    return log_py_theta_star_1/log_py_theta_star_2




def DIC_comparison(path_1, path_2):

    _, _, log_prob_1, log_prob_2 = get_maps(path_1, path_2)

    idx_map_1 = np.argmax(log_prob_1)
    idx_map_2 = np.argmax(log_prob_2)

    log_py_theta_star_1 = log_prob_1[idx_map_1]
    log_py_theta_star_2 = log_prob_2[idx_map_2]

    mean_log_py_1 = np.mean(log_prob_1)
    mean_log_py_2 = np.mean(log_prob_2)

    p_D_1 = 2 * (log_py_theta_star_1 - mean_log_py_1)
    p_D_2 = 2 * (log_py_theta_star_2 - mean_log_py_2)

    DIC_1 = -2 * (log_py_theta_star_1 - p_D_1)
    DIC_2 = -2 * (log_py_theta_star_2 - p_D_2)


    return DIC_1/DIC_2



path1 = '/Users/jacopobottoni/Bayesian_project/10000_4step_try1_step_predefiniti.h5'
path2 = '/Users/jacopobottoni/Bayesian_project/30000_4step_try1_step_predefiniti.h5'
path3 = '/Users/jacopobottoni/Bayesian_project/100000_5step_try4_step_predefiniti.h5'

print(bayes_ratio(path1, path2))
print(bayes_ratio(path1, path3))
print(bayes_ratio(path2, path3))

print(DIC_comparison(path1, path2))
print(DIC_comparison(path1, path3))
print(DIC_comparison(path2, path3))

