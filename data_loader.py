# BEFORE
# @lru_cache(maxsize=32)
# def load_dataset_cached(name, frames_dict_serialized=None):

from functools import lru_cache
import os
import pandas as pd

DATA_DIR = "data"

def list_datasets():
    base = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    return base

def load_all_datasets():
    frames = {}
    for f in list_datasets():
        path = os.path.join(DATA_DIR, f)
        try:
            frames[f] = pd.read_csv(path)
        except Exception:
            pass
    return frames

@lru_cache(maxsize=32)
def load_dataset_cached(name: str):
    """Cache only by dataset name on disk."""
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path)
