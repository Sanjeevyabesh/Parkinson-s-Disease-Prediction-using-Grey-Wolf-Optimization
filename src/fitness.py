from src.model import train_svm, evaluate
import numpy as np


def fitness_function(solution, X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]
    
    # Split solution
    feature_mask = solution[:n_features] > 0.5
    C = abs(solution[-2])
    gamma = abs(solution[-1])
    
    # Avoid empty features
    if sum(feature_mask) == 0:
        return 1
    
    X_train_sel = X_train[:, feature_mask]
    X_test_sel = X_test[:, feature_mask]
    
    model = train_svm(X_train_sel, y_train, C, gamma)
    acc = evaluate(model, X_test_sel, y_test)
    
    # Fitness (minimize)
    feature_ratio = sum(feature_mask) / n_features
    fitness = (1 - acc) + 0.2 * feature_ratio
    
    return fitness
