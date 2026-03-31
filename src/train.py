import numpy as np
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.fitness import fitness_function
from src.gwo import gwo
from src.model import train_svm, evaluate
import matplotlib.pyplot as plt
import os
from src.model import detailed_metrics



def run_training(data_path):
    
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    dim = X_train.shape[1] + 2  # features + C + gamma
    
    best_solution, convergence = gwo(
        fitness_function,
        dim=dim,
        lb=0,
        ub=1,
        max_iter=20,
        pop_size=10,
        args=(X_train, X_test, y_train, y_test)
    )
    
    # Extract solution
    feature_mask = best_solution[:X_train.shape[1]] > 0.5
    C = abs(best_solution[-2])
    gamma = abs(best_solution[-1])
    
    X_train_sel = X_train[:, feature_mask]
    X_test_sel = X_test[:, feature_mask]
    
    model = train_svm(X_train_sel, y_train, C, gamma)
    acc = evaluate(model, X_test_sel, y_test)
    
    print("Final Accuracy:", acc)

    os.makedirs("results", exist_ok=True)

        # Baseline model
    base_model = train_svm(X_train, y_train, C=1, gamma=0.1)
    base_acc = evaluate(base_model, X_test, y_test)

    print("Baseline Accuracy:", base_acc)

    detailed_metrics(model, X_test_sel, y_test)

    feature_mask = best_solution[:X_train.shape[1]] > 0.5

    print("\nTotal Features:", X_train.shape[1])
    print("Selected Features:", sum(feature_mask))

    # Plot convergence
    plt.plot(convergence)
    plt.title("GWO Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("results/convergence.png")
    plt.show()