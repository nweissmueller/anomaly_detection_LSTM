import pandas as pd
from src.utils import create_dataset


def load_and_prepare_data(filepath):
    with open(filepath) as f:
        df = pd.read_csv(f)
        df = df.sample(frac=1.0)
        df = df.drop(labels=['id', 'labtest'], axis=1)
    return df
