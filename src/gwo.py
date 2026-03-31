import numpy as np


def gwo(opt_func, dim, lb, ub, max_iter, pop_size, args):
    
    alpha = np.zeros(dim)
    beta = np.zeros(dim)
    delta = np.zeros(dim)
    
    alpha_score = float('inf')
    beta_score = float('inf')
    delta_score = float('inf')
    
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    
    convergence = []
    
    for t in range(max_iter):
        for i in range(pop_size):
            
            fitness = opt_func(positions[i], *args)
            
            if fitness < alpha_score:
                alpha_score = fitness
                alpha = positions[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta = positions[i].copy()
        
        a = 2 - t * (2 / max_iter)
        
        for i in range(pop_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = abs(C1 * alpha[j] - positions[i][j])
                X1 = alpha[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                D_beta = abs(C2 * beta[j] - positions[i][j])
                X2 = beta[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = abs(C3 * delta[j] - positions[i][j])
                X3 = delta[j] - A3 * D_delta
                
                positions[i][j] = (X1 + X2 + X3) / 3
        
        convergence.append(alpha_score)
        print(f"Iteration {t+1}, Best Fitness: {alpha_score}")
    
    return alpha, convergence
