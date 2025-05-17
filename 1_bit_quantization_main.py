import numpy as np
from scipy.stats import norm
import yaml
import func_definitions as fd
from numpy.random import default_rng
from concurrent.futures import ThreadPoolExecutor
import threading

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():

    config = load_config()
    sim = config["simulation"]
    
    rng = default_rng(sim["seed"])

    # Initialize CEM parameters
    CEM_param = sim["CEM"]
    N = CEM_param["N"]
    alpha = CEM_param["alpha"]
    elite = CEM_param["elite"]

    data = sim["data"]
    std = data["std"]
    mu = np.zeros(2)
    mu[0] = std
    mu[1] = -1*std
    sig = np.zeros(2)
    sig[0] = 1
    sig[1] = 1
    
    # Generate Data
    s = data["s"]
    mean = data["mean"]
    X = norm.rvs(loc=mean, scale=std,size=s, random_state=rng)
    
    # Evaluate Risk Measure for initial mu, initialize optimal solution
    h = sim["h"]
    option_h = h["option_h"]
    param_h = h["param_h"]
    d = sim["d"]
    option_d = d["name"]
    f_min = fd.risk_measure(mu,X,option_d=option_d,param_d=None,option_h=option_h,param_h=param_h)
    print(f"f_min: {f_min}")
    opt_sol = np.zeros(2)
    opt_sol[0] = mu[0]
    opt_sol[1] = mu[1]

    tmax = sim["tmax"]
    t = 0

    while t <= tmax:
        t += 1 
        print(t)
        samples = np.column_stack([
            norm.rvs(loc=mu[0], scale=sig[0], size=N),
            norm.rvs(loc=mu[1], scale=sig[1], size=N)
        ])

        with ThreadPoolExecutor() as executor:
            f_eval = list(executor.map(lambda sample: fd.risk_measure(sample, X, option_d=option_d,param_d=None,option_h=option_h,param_h=param_h), samples))
            print(f"Number of active threads: {len(threading.enumerate())}")
            print(f"f_eval: {f_eval}")
        f_eval = np.array(f_eval)

        good_indices = np.where(f_eval < f_min)[0]

        better_indices = np.where(f_eval < f_min)[0]
        if better_indices.size > 0:
            best_idx = better_indices[np.argmin(f_eval[better_indices])]
            opt_sol[:] = samples[best_idx]
            f_min = f_eval[best_idx]
        print("opt solution so far")
        print(opt_sol)
        print("f_min")
        print(f_min)

        if good_indices.size > 0:
            elite_indices = good_indices[np.argsort(f_eval[good_indices])[:elite]]
            elite_samples = samples[elite_indices]
        else:
           elite_samples[:] = opt_sol  # Handle empty case

        mu_update = np.mean(elite_samples, axis=0)
        sig_update2 = np.mean((elite_samples - mu_update) ** 2, axis=0)

        #  smoothing
        mu[:] = (1 - alpha) * mu + alpha * mu_update
        sig[:] = (1 - alpha) * sig + alpha * np.sqrt(sig_update2)

        print("current mu")
        print(mu)
        print("current sig")
        print(sig)

    print(opt_sol)
    print(f_min)
    print(fd.risk_measure(opt_sol,X,option_d=option_d,param_d=None,option_h=option_h,param_h=param_h))

if __name__ == "__main__":
    main()