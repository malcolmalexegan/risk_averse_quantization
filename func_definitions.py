import numpy as np
from scipy.integrate import quad_vec

def h_power(w,k):
    return np.power(w,k)

def d_SE(y,X):
    return np.minimum((X - y[0])**2, (X - y[1])**2)

def d_SE_weight(y,X,exponent):
    return np.minimum(np.pow(np.abs(X),exponent)*(X - y[0])**2, np.abs(X)*(X - y[1])**2)

def prob_est(u, y, X,option_d,param_d):
    if option_d == "SE":
        prob = np.mean(d_SE(y,X) > u)
    elif option_d == "SE_weight":
        prob = np.mean(d_SE_weight(y,X,param_d) > u)
    return prob

def risk_measure(y,X,option_d,param_d,option_h,param_h):
    if option_d == "SE" and option_h == "Power":
        integral, error = quad_vec(lambda u: h_power(prob_est(u, y, X, "SE",param_d=None),param_h), 0, np.inf)
    elif option_d == "SE_weight" and option_h == "Power":
        integral, error = quad_vec(lambda u: h_power(prob_est(u, y, X, "SE_weight", param_d),param_h), 0, np.inf)
    return integral
