import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train import run_training

if __name__ == "__main__":
    run_training("data/parkinsons.csv")