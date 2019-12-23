import os
import pickle

import pandas as pd

import numpy as np

import joblib

import pathlib

from sklearn.preprocessing import StandardScaler


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

BIN_PATH = PATH / "bin"

DATA_PATH = PATH / "_data"

COLUMNS_TO_REMOVE = [
    'scls_h', 'scls_j', 'scls_k',
    "AndersonDarling", "AmplitudeJ", "AmplitudeH", "AmplitudeJH", "AmplitudeJK",
    'Freq1_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_0',
    "CAR_mean", "CAR_tau", "CAR_sigma"] 

COLUMNS_NO_FEATURES = ['id', 'tile', 'cnt', 'ra_k', 'dec_k', 'vs_type', 'vs_catalog', 'cls'] 


def create_dir(path):
    path = pathlib.Path(path)
    if path.is_dir():
        raise IOError("Please remove the directory {}".format(path))
    os.makedirs(str(path))
    
    
def read_original_data():
    print("Reading original files...")
    with joblib.Parallel(backend="multiprocessing") as P:
        df = pd.concat(
            P(joblib.delayed(pd.read_pickle)(fn)
             for fn in BIN_PATH.glob("*.pkl.bz2"))
        )
    return df


def clean(df):
    print("Removing bad rows")

    df = df.dropna()

    df = df[df.cnt >= 30]

    df = df[
        df.c89_hk_color.between(-100, 100) &
        df.c89_jh_color.between(-100, 100) &
        df.c89_jk_color.between(-100, 100) &
        df.n09_hk_color.between(-100, 100) &
        df.n09_jh_color.between(-100, 100) &
        df.n09_jk_color.between(-100, 100)]

    df = df[~np.isinf(df.Period_fit.values)]
    df = df[~df.Gskew.isnull()]

    print("Removing unused columns")

    df = df[[c for c in df.columns if c not in COLUMNS_TO_REMOVE]]
    return df


def reorder(df):
    print("Reordering")
    features = [c for c in df.columns.values if c not in COLUMNS_NO_FEATURES]
    order = COLUMNS_NO_FEATURES + features 
    return df[order]


def to_int32(df):
    print("To int32")
    df = df.copy()
    features = [c for c in df.columns if c not in COLUMNS_NO_FEATURES]
    df[features] = df[features].astype(np.float32)
    return df


def scale(df):
    print("Scaling")
    df = df.copy()
    features = [c for c in df.columns.values if c not in COLUMNS_NO_FEATURES]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features].values)
    
    return scaler, df


def sample(df, n):
    total_size = int(df.groupby("tile").id.count().mean())
    print("Subsampling unknown of {} by tile".format(n))
    pcls = [df[df.cls == 1].copy()]
    ncls = [
         tdf.sample(n, random_state=42).copy()
         for t, tdf in df[df.cls == 0].groupby("tile")]
    return pd.concat(pcls + ncls, ignore_index=True)


def store(obj, fname):
    print("Storing {}...".format(fname))
    store_path = DATA_PATH / fname
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(store_path, compression="bz2")
    else:
        with open(str(store_path), "wb") as fp:
            pickle.dump(obj, fp)

            
def build():
    create_dir(DATA_PATH)
    s20k = read_original_data()
    s20k = clean(s20k)
    s20k = reorder(s20k)
    s20k = to_int32(s20k)  
    store(s20k, "s20k.pkl.bz2")

    s20k_scaler, s20k_scaled = scale(s20k)
    store(s20k_scaler, "scaler_s20k.pkl")
    store(s20k_scaled, "s20k_scaled.pkl.bz2")

    s5k = sample(s20k, 5000)
    store(s5k, "s5k.pkl.bz2")

    s5k_scaler, s5k_scaled = scale(s5k)
    store(s5k_scaler, "scaler_s5k.pkl")
    store(s5k_scaled, "s5k_scaled.pkl.bz2")

    s2_5k = sample(s5k, 2500)
    store(s2_5k, "s2_5k.pkl.bz2")

    s2_5k_scaler, s2_5k_scaled = scale(s2_5k)
    store(s2_5k_scaler, "scaler_s2_5k.pkl")
    store(s2_5k_scaled, "s2_5k_scaled.pkl.bz2")
    
    
if __name__ == "__main__":
    build()