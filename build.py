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

CACHE_PATH = PATH / "_cache"


COLUMNS_TO_REMOVE = [
    'scls_h', 'scls_j', 'scls_k',
    "AndersonDarling", "AmplitudeJ", "AmplitudeH", "AmplitudeJH", "AmplitudeJK",
    'Freq1_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_0',
    "CAR_mean", "CAR_tau", "CAR_sigma", "StetsonK", "Meanvariance"] 

COLUMNS_NO_FEATURES = ['id', 'tile', 'cnt', 'ra_k', 'dec_k', 'vs_type', 'vs_catalog', 'cls']

FULL_DATA_PATH = "/home/jbcabral/carpyncho3/production_data/stored/light_curves/{}/features_{}.npy"


SAMPLES = joblib.load(BIN_PATH / "sampleids.pkl")

SAMPLES_2 = joblib.load(BIN_PATH / "sampleids2.pkl")

SAMPLES_3 = joblib.load(BIN_PATH / "sampleids3.pkl")


def create_dir(path):
    path = pathlib.Path(path)
    if path.is_dir():
        raise IOError("Please remove the directory {}".format(path))
    os.makedirs(str(path))

    
def clean(df):
    df = df[[c for c in df.columns if c not in COLUMNS_TO_REMOVE]]
    
    # clean
    df = df.dropna()

    df = df[
        (df.cnt >= 30) &
        df.Mean.between(12, 16.5, inclusive=False) &
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


# def _read_original_parallel(path):
#     df = pd.read_pickle(path)
#     df = clean(df)
#     return df

# def read_original_data():
#     print("Reading original files...")
#     with joblib.Parallel(backend="multiprocessing") as P:
#         parts = P(
#             joblib.delayed(_read_original_parallel)(fn)
#             for fn in BIN_PATH.glob("*.pkl.bz2"))
#     df = pd.concat(parts, ignore_index=True)                                            
#     return df


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


def sample(df, n, seeds):
    ids = seeds[n]
    return df[df.id.isin(ids)].copy()


def store(obj, fname, subfolder):
    print("Storing {}...".format(fname))
    if subfolder is None:
        store_path = DATA_PATH / fname
    else:
        store_path = DATA_PATH / subfolder / fname
    
    if not store_path.parent.is_dir():
        store_path.parent.mkdir(parents=True)
        
    if isinstance(obj, pd.DataFrame):
        obj.to_pickle(store_path, compression="bz2")
    else:
        with open(str(store_path), "wb") as fp:
            pickle.dump(obj, fp)


def _build_samples(full, seeds, subfolder):
    # 10 %
    s10p = sample(full, 0.1, seeds=seeds)
    store(s10p, "s10p.pkl.bz2", subfolder=subfolder)
    
    s10p_scaler, s10p_scaled = scale(s10p)
    store(s10p_scaler, "scaler_s10p.pkl", subfolder=subfolder)
    store(s10p_scaled, "s10p_scaled.pkl.bz2", subfolder=subfolder)
    
    
    # 20 mil
    s20k = sample(s10p, 20000, seeds=seeds)
    store(s20k, "s20k.pkl.bz2", subfolder=subfolder)
    
    s20k_scaler, s20k_scaled = scale(s20k)
    store(s20k_scaler, "scaler_s20k.pkl", subfolder=subfolder)
    store(s20k_scaled, "s20k_scaled.pkl.bz2", subfolder=subfolder)
    
    # 5 mil
    
    s5k = sample(s20k, 5000, seeds=seeds)
    store(s5k, "s5k.pkl.bz2", subfolder=subfolder)

    s5k_scaler, s5k_scaled = scale(s5k)
    store(s5k_scaler, "scaler_s5k.pkl", subfolder=subfolder)
    store(s5k_scaled, "s5k_scaled.pkl.bz2", subfolder=subfolder)
    
    # 2500
    
    s2_5k = sample(s5k, 2500, seeds=seeds)
    store(s2_5k, "s2_5k.pkl.bz2", subfolder=subfolder)

    s2_5k_scaler, s2_5k_scaled = scale(s2_5k)
    store(s2_5k_scaler, "scaler_s2_5k.pkl", subfolder=subfolder)
    store(s2_5k_scaled, "s2_5k_scaled.pkl.bz2", subfolder=subfolder)
    
    # uno a uno
    
    sO2O = sample(s2_5k, "O2O", seeds=seeds)
    store(sO2O, "sO2O.pkl.bz2", subfolder=subfolder)

    sO2O_scaler, sO2O_scaled = scale(sO2O)
    store(sO2O_scaler, "scaler_sO2O.pkl", subfolder=subfolder)
    store(sO2O_scaled, "sO2O_scaled.pkl.bz2", subfolder=subfolder)
    
            
            
def build():
#     if DATA_PATH.is_dir():
#         raise IOError(f"Please remove the directory {DATA_PATH}")
#     if CACHE_PATH.is_dir():
#         raise IOError(f"Please remove the directory {CACHE_PATH}")
    
    # full
    
    full = read_full_data([
        'b206', 'b214', 'b216', 'b220', 'b228', 'b234', 'b247', 'b248', 
        'b261', 'b262', 'b263', 'b264', 'b277', 'b278', 'b360', 'b396'])
    
    store(full, "full.pkl.bz2", subfolder=None)
    
    full_scaler, full_scaled = scale(full)
    store(full_scaler, "scaler_full.pkl", subfolder=None)
    store(full_scaled, "full_scaled.pkl.bz2", subfolder=None)
    
#     # Sampling
    
#     print(">>> Sample 1/3")
#     _build_samples(full, seeds=SAMPLES, subfolder=None)
    
#     print(">>> Sample 2/3")
#     _build_samples(full, seeds=SAMPLES_2, subfolder="data_2")
    
#     print(">>> Sample 3/3")
#     _build_samples(full, seeds=SAMPLES_3, subfolder="data_3")
    
    
    
    

    
    
    
    
if __name__ == "__main__":
    build()