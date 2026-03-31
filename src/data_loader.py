import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    
    # Drop name column if exists
    if 'name' in df.columns:
        df = df.drop(['name'], axis=1)
    
    X = df.drop('status', axis=1)
    y = df['status']
    
    return X, y
