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

FULL_DATA_PATH = "/home/jbcabral/carpyncho3/production_data/stored/light_curves/{}/features_{}.npy"


with open(BIN_PATH / "sampleids.pkl", "rb") as fp:
    SAMPLES = pickle.load(fp)


def create_dir(path):
    path = pathlib.Path(path)
    if path.is_dir():
        raise IOError("Please remove the directory {}".format(path))
    os.makedirs(str(path))

    
def clean(df):
    df = df[[c for c in df.columns if c not in COLUMNS_TO_REMOVE]]
    
    # clean
    df = df.dropna()

    df = df[df.cnt >= 30]

    df = df[
        df.c89_hk_color.between(-100, 100) &
        df.c89_jh_color.between(-100, 100) &
        df.c89_jk_color.between(-100, 100) &
        df.n09_hk_color.between(-100, 100) &
        df.n09_jh_color.between(-100, 100) &
        df.n09_jk_color.between(-100, 100)]
    
    # features columns
    features = [c for c in df.columns.values if c not in COLUMNS_NO_FEATURES]
    features.sort()
    
    # to float32
    df[features] = df[features].astype(np.float32)
    
    # reorder
    order = COLUMNS_NO_FEATURES + features 
    df = df[order]
    
    df = df[~np.isinf(df.Period_fit.values)]
    df = df[~df.Gskew.isnull()]
    return df


def _read_original_parallel(path):
    df = pd.read_pickle(path)
    df = clean(df)
    return df

def read_original_data():
    print("Reading original files...")
    with joblib.Parallel(backend="multiprocessing") as P:
        parts = P(
            joblib.delayed(_read_original_parallel)(fn)
            for fn in BIN_PATH.glob("*.pkl.bz2"))
    df = pd.concat(parts, ignore_index=True)                                            
    return df



def _read_full_parallel(tile):
    df_path = FULL_DATA_PATH.format(tile, tile)
    df = pd.DataFrame(np.load(df_path))
    
    df["tile"] = tile
    df["vs_type"] = df.vs_type.str.decode("utf8")
    df = df[(df.vs_type == "") | df.vs_type.str.startswith("RRLyr-")]
    df["cls"] = df.vs_type.apply(lambda  vst: 0 if vst == "" else 1)
    
    df = clean(df)
    return df

def read_full_data(tiles):
    print("Reading full files...")
    with joblib.Parallel(backend="multiprocessing") as P:
        parts = P(
            joblib.delayed(_read_full_parallel)(tile)
            for tile in tiles)
    df = pd.concat(parts, ignore_index=True)                                            
    return df
    

    

def scale(df):
    print("Scaling")
    df = df.copy()
    features = [c for c in df.columns.values if c not in COLUMNS_NO_FEATURES]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features].values)
    
    return scaler, df


def sample(df, n):
    ids = SAMPLES[n]
    return df[df.id.isin(ids)].copy()


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
    
    sO2O = sample(s2_5k, "O2O")
    store(sO2O, "sO2O.pkl.bz2")

    sO2O_scaler, sO2O_scaled = scale(sO2O)
    store(sO2O_scaler, "scaler_sO2O.pkl")
    store(sO2O_scaled, "sO2O_scaled.pkl.bz2")
    
    full = read_full_data(['b234', 'b360', 'b278', 'b261'])
    store(full, "full.pkl.bz2")
    
    full_scaler, full_scaled = scale(full)
    store(full_scaler, "scaler_full.pkl")
    store(full_scaled, "full_scaled.pkl.bz2")
    
    
if __name__ == "__main__":
    build()