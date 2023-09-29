import numpy as np
import pandas as pd




# Scenario I: the post-nonlinear model for test
funcs = {
    "linear": lambda x: x,
    "square": lambda x: x**2,
    "cos": lambda x: np.cos(x),
    "cube": lambda x: x**3,
    "tanh": lambda x: np.tanh(x),
}


func_names = ["linear", "square", "cos", "cube", "tanh"]

def data_gen(n_samples, dim, test_type, noise="gaussian"):
    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    keys = np.random.choice(range(5), 2)
    pnl_funcs = [func_names[k] for k in keys]

    func1 = funcs[pnl_funcs[0]]
    func2 = funcs[pnl_funcs[1]]

    x = 0.5 * sampler(size=(n_samples, 1))   
    y = 0.5 * sampler(size=(n_samples, 1))   
    z = sampler(size=(n_samples, dim))
    m = np.mean(z, axis=1).reshape(-1, 1)
    x += m
    y += m
    x, y = func1(x), func2(y)

    if test_type:
        return x, y, z
    else:
        eb = 0.5 * sampler(size=(n_samples, 1))  
        x += eb
        y += eb
        return x, y, z


def generate_Scetbon_data(test_type, num_samples, dim, noise="gaussian", seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)
    
    x, y, z = data_gen(n_samples=num_samples, test_type=test_type, dim=dim, noise=noise)
    
    return np.array(x), np.array(y), np.array(z)





# Scenario II: the heavy tailed model for test
def heavy_tailed_data_generate(size=500, dz=5, sType='CI', seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)
        
    n = size
    e1 = np.random.standard_t(1, (n, 1))
    e2 = np.random.standard_t(1, (n, 1))
    z = np.random.normal(0., 1., (n, dz))
    z_mean = np.mean(z, axis=1).reshape(-1, 1)
    
    if sType == 'CI':
        x = z_mean + e1
        y = z_mean + e2
        return np.array(x), np.array(y), np.array(z)
    else:
        x = z_mean + e1
        y = z_mean + e1 + e2
        return np.array(x), np.array(y), np.array(z)





# Scenario III: the chain structure for test
def chain_data(size=500, Type='CI', dz=5, seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    num = size
    Y = np.random.normal(1., 1., (num, 1))
    a = np.random.uniform(0., 0.3, (1, dz))
    b = np.random.uniform(0., 0.3, (dz, 1))

    if Type == 'CI':
        Z = np.matmul(Y,a) + np.random.normal(0., 1., (num, dz))
        X = np.matmul(Z,b) + np.random.normal(0., 1., (num, 1))
    else:
        Z = np.matmul(Y,a) + np.random.normal(0., 1., (num, dz))
        X = np.matmul(Z,b) + Y + np.random.normal(0., 1., (num, 1))

    return np.array(X), np.array(Y), np.array(Z)





# Scenario IV: parameter selection model
def best_parameter_data(size=500, sType='CI', dz=5, dist='gaussian', seed=None):
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)
    
    n = size
    if dist == 'gaussian':
        z = np.random.normal(0, 1, (n, dz))
        ex = np.random.normal(0, 1, (n, 1))
        ey = np.random.normal(0, 1, (n, 1))
    elif dist == 'uniform':
        z = np.random.uniform(-1, 1, (n, dz))
        ex = np.random.uniform(-1, 1, (n, 1))
        ey = np.random.uniform(-1, 1, (n, 1))
    
    alpha = np.random.uniform(0, 2)

    if sType == 'CI':
        x = ex
        y = ey
        return np.array(x), np.array(y), np.array(z)
    else:
        x = ex
        y = alpha * x + 0.5 * ey
        return np.array(x), np.array(y), np.array(z)