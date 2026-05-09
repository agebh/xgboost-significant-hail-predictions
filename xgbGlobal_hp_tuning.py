#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
from pathlib import Path
from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import psutil
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from netCDF4 import Dataset

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
)

try:
    from optuna_integration.xgboost import XGBoostPruningCallback
except ModuleNotFoundError:
    from optuna.integration import XGBoostPruningCallback


# ============================================================
# Thread settings
# ============================================================
gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
n_jobs = 2
threads_per_trial = 1

print("gpus:", gpus, "n_jobs:", n_jobs, "threads_per_trial:", threads_per_trial)
print("SLURM_MEM_PER_NODE:", os.environ.get("SLURM_MEM_PER_NODE"))
print("SLURM_MEM_PER_CPU :", os.environ.get("SLURM_MEM_PER_CPU"))
print("RAM total (GB):", psutil.virtual_memory().total / 1e9)

os.environ["OMP_NUM_THREADS"] = str(threads_per_trial)
os.environ["MKL_NUM_THREADS"] = str(threads_per_trial)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_trial)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_trial)


def main():
    # ============================================================
    # 1) Settings / constants
    # ============================================================
    start_day = datetime(2000, 1, 1, 0)
    stop_day = datetime(2022, 12, 31, 23)
    missing_year = 2003

    threshold_cm = 4.4
    ds_factor_us = 39
    ds_factor_eu = 290

    run_date = "xgbGlobal_sigHail/20260411_tuning_round2"
    
    # Paths
    noaa_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/hail_observations/SPC_data_griddedERA")
    essl_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/hail_observations/ESSL_data_griddedERA")

    era_conus_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/e5_hailpredictors_conus")
    era_eu_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/e5_hailpredictors_eu")
    era_invar_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/e5_data_processing/e5_invariant")

    era_const_fields = era_invar_dir / "e5_invariant_129_z_ll025sc.2020010100_2020010100.nc"
    lsm_file = era_invar_dir / "e5_lsm_11024sc.nc"

    # Bounding boxes
    us_lat_min, us_lat_max = 25.25, 49.0
    us_lon_min, us_lon_max = -130.0, -65.0

    eu_lat_min, eu_lat_max = 35.0, 65.0
    eu_lon_min, eu_lon_max = -10.0, 35.0

    # Predictors
    all_model_vars = [
        "CAPEmax", "SRH03", "VS03", "FLH",
        "CINmax", "SRH06", "VS06", "DewT",
        "TotalTotals", "RH850", "RH500",
    ]

    model_vars = [
        "CAPEmax", "SRH03", "VS03", "FLH",
        "CINmax", "VS06", "DewT",
        "TotalTotals",
    ]

    feature_names = list(model_vars)
    ix_vs03 = model_vars.index("VS03")
    ix_vs06 = model_vars.index("VS06")
    sel_idx = [all_model_vars.index(v) for v in model_vars]

    print("Selected predictor indices:", sel_idx, "=>", [all_model_vars[i] for i in sel_idx])

    # Output folders
    models_dir = Path("/cluster/home/agebhardt/models") / run_date
    preds_dir = Path("/cluster/home/agebhardt/predictions") / run_date
    stats_dir = Path("/cluster/home/agebhardt/statistics") / run_date
    plots_dir = Path("/cluster/home/agebhardt/plots") / run_date

    for d in (models_dir, preds_dir, stats_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 2) Time vectors
    # ============================================================
    rgd_time_dd = pd.date_range(start_day, end=stop_day, freq="D")
    rgd_time_dd = rgd_time_dd[rgd_time_dd.year != missing_year]
    years_all = np.unique(rgd_time_dd.year)

    # ============================================================
    # 3) Read ERA invariant fields
    # ============================================================
    with Dataset(era_const_fields, mode="r") as nc:
        rgr_lat = np.squeeze(nc.variables["latitude"][:]).astype(np.float32)
        rgr_lon = np.squeeze(nc.variables["longitude"][:]).astype(np.float32)

    rgr_lon = np.where(rgr_lon > 180, rgr_lon - 360, rgr_lon)

    print("ERA invariant fields read.")

    # ============================================================
    # 4) NOAA observations
    # ============================================================
    hail_vars = ["Hail", "HailSize"]

    idx_lat_noaa = None
    idx_lon_noaa = None
    rgr_noaa_obs = None

    dd = 0
    for y in years_all:
        nc_path = noaa_dir / f"SPC-Hail-StormReports_gridded-75km_{int(y)}.nc"
        with Dataset(str(nc_path), mode="r") as nc:
            lat1d_full = np.squeeze(nc.variables["lat"][:, 0]).astype(np.float32)
            lon1d_full = np.squeeze(nc.variables["lon"][0, :]).astype(np.float32)

            if idx_lat_noaa is None:
                idx_lat_noaa = np.where((lat1d_full >= us_lat_min) & (lat1d_full <= us_lat_max))[0]
                idx_lon_noaa = np.where((lon1d_full >= us_lon_min) & (lon1d_full <= us_lon_max))[0]

                if idx_lat_noaa.size == 0 or idx_lon_noaa.size == 0:
                    raise RuntimeError("NOAA bbox produced empty subset.")

                ny_noaa = idx_lat_noaa.size
                nx_noaa = idx_lon_noaa.size
                rgr_noaa_obs = np.zeros((2, len(rgd_time_dd), ny_noaa, nx_noaa), dtype=np.float32)

                print("NOAA cropped shape:", ny_noaa, nx_noaa)

            yearlength = 365 + int(calendar.isleap(int(y)))

            for ii, vname in enumerate(hail_vars):
                data_full = np.squeeze(nc.variables[vname][:]).astype(np.float32)
                rgr_noaa_obs[ii, dd:dd + yearlength, :, :] = data_full[:, idx_lat_noaa, :][:, :, idx_lon_noaa]

        dd += yearlength

    print("rgr_noaa_obs shape:", rgr_noaa_obs.shape)

    # ============================================================
    # 5) ESSL observations
    # ============================================================
    size_name_candidates = ["HailSize", "size", "Size"]

    idx_lat_essl = None
    idx_lon_essl = None
    rgr_essl_obs = None

    dd = 0
    for y in years_all:
        nc_path = essl_dir / f"ESSL-Hail-StormReports_gridded-ERA5_{int(y)}.nc"

        with Dataset(str(nc_path), mode="r") as nc:
            if idx_lat_essl is None:
                lat1d_full = np.squeeze(nc.variables["lat"][:, 0]).astype(np.float32)
                lon1d_full = np.squeeze(nc.variables["lon"][0, :]).astype(np.float32)
                lon1d_full = np.where(lon1d_full > 180, lon1d_full - 360, lon1d_full)

                idx_lat_essl = np.where((lat1d_full >= eu_lat_min) & (lat1d_full <= eu_lat_max))[0]
                idx_lon_essl = np.where((lon1d_full >= eu_lon_min) & (lon1d_full <= eu_lon_max))[0]

                if idx_lat_essl.size == 0 or idx_lon_essl.size == 0:
                    raise RuntimeError("EU bbox produced empty subset for ESSL obs.")

                size_varname = None
                for cand in size_name_candidates:
                    if cand in nc.variables:
                        size_varname = cand
                        break
                if size_varname is None:
                    raise KeyError(f"No hail size variable found in {nc_path.name}")

                ny_eu = idx_lat_essl.size
                nx_eu = idx_lon_essl.size
                rgr_essl_obs = np.zeros((2, len(rgd_time_dd), ny_eu, nx_eu), dtype=np.float32)

                print("ESSL size variable detected:", size_varname)
                print("ESSL cropped shape:", ny_eu, nx_eu)

            yearlength = 365 + int(calendar.isleap(int(y)))

            hail_full = np.squeeze(nc.variables["Hail"][:]).astype(np.float32)
            hail_crop = hail_full[:, idx_lat_essl, :][:, :, idx_lon_essl]

            size_full = np.squeeze(nc.variables[size_varname][:]).astype(np.float32)
            size_crop = size_full[:, idx_lat_essl, :][:, :, idx_lon_essl]

            rgr_essl_obs[0, dd:dd + yearlength, :, :] = hail_crop
            rgr_essl_obs[1, dd:dd + yearlength, :, :] = size_crop

        dd += yearlength

    print("rgr_essl_obs shape:", rgr_essl_obs.shape)

    # ============================================================
    # 6) Predictor grids US / EU
    # ============================================================
    sample_conus = None
    for y in years_all:
        p = era_conus_dir / f"ERA5_new_predictors_conus_{int(y)}.npz"
        if p.is_file():
            sample_conus = np.load(p)
            break
    if sample_conus is None:
        raise FileNotFoundError("Could not find any CONUS predictor file")

    lat_us_full = sample_conus["rgrLat"].astype(np.float32)
    lon_us_full = sample_conus["rgrLon"].astype(np.float32)
    lon_us_full_m180 = np.where(lon_us_full > 180, lon_us_full - 360, lon_us_full)

    idx_lat_us = np.where((lat_us_full >= us_lat_min) & (lat_us_full <= us_lat_max))[0]
    idx_lon_us = np.where((lon_us_full_m180 >= us_lon_min) & (lon_us_full_m180 <= us_lon_max))[0]

    ny_us, nx_us = idx_lat_us.size, idx_lon_us.size
    rgr_era_varall_us = np.zeros((len(rgd_time_dd), len(model_vars), ny_us, nx_us), dtype=np.float32)

    iyear = 0
    for y in years_all:
        infile = era_conus_dir / f"ERA5_new_predictors_conus_{int(y)}.npz"
        data_tmp = np.load(infile)
        rgr_vars = data_tmp["rgrERAVarsyy"].astype(np.float32)[:, sel_idx, :, :]
        rgr_vars = rgr_vars[:, :, idx_lat_us, :][:, :, :, idx_lon_us]
        yearlength = rgr_vars.shape[0]
        rgr_era_varall_us[iyear:iyear + yearlength, :, :, :] = rgr_vars
        iyear += yearlength

    print("rgr_era_varall_us shape:", rgr_era_varall_us.shape)

    sample_eu = None
    for y in years_all:
        p = era_eu_dir / f"ERA5_new_predictors_eu_{int(y)}.npz"
        if p.is_file():
            sample_eu = np.load(p)
            break
    if sample_eu is None:
        raise FileNotFoundError("Could not find any EU predictor file")

    lat_eu_full = sample_eu["rgrLat"].astype(np.float32)
    lon_eu_full = sample_eu["rgrLon"].astype(np.float32)
    lon_eu_full_m180 = np.where(lon_eu_full > 180, lon_eu_full - 360, lon_eu_full)

    idx_lat_eu = np.where((lat_eu_full >= eu_lat_min) & (lat_eu_full <= eu_lat_max))[0]
    idx_lon_eu = np.where((lon_eu_full_m180 >= eu_lon_min) & (lon_eu_full_m180 <= eu_lon_max))[0]

    ny_eu, nx_eu = idx_lat_eu.size, idx_lon_eu.size
    rgr_era_varall_eu = np.zeros((len(rgd_time_dd), len(model_vars), ny_eu, nx_eu), dtype=np.float32)

    iyear = 0
    for y in years_all:
        infile = era_eu_dir / f"ERA5_new_predictors_eu_{int(y)}.npz"
        data_tmp = np.load(infile)
        rgr_vars = data_tmp["rgrERAVarsyy"].astype(np.float32)[:, sel_idx, :, :]
        rgr_vars = rgr_vars[:, :, idx_lat_eu, :][:, :, :, idx_lon_eu]
        yearlength = rgr_vars.shape[0]
        rgr_era_varall_eu[iyear:iyear + yearlength, :, :, :] = rgr_vars
        iyear += yearlength

    print("rgr_era_varall_eu shape:", rgr_era_varall_eu.shape)

    # ============================================================
    # 7) Land-sea masks
    # ============================================================
    with Dataset(lsm_file, mode="r") as nc:
        lsm = np.squeeze(nc.variables["lsm"][:]).astype(np.float32)

    idx_lat_glob_us = np.where((rgr_lat >= us_lat_min) & (rgr_lat <= us_lat_max))[0]
    idx_lon_glob_us = np.where((rgr_lon >= us_lon_min) & (rgr_lon <= us_lon_max))[0]
    lsm_us = lsm[idx_lat_glob_us][:, idx_lon_glob_us]

    idx_lat_glob_eu = np.where((rgr_lat >= eu_lat_min) & (rgr_lat <= eu_lat_max))[0]
    idx_lon_glob_eu = np.where((rgr_lon >= eu_lon_min) & (rgr_lon <= eu_lon_max))[0]
    lsm_eu = lsm[idx_lat_glob_eu][:, idx_lon_glob_eu]

    print("LSM_US shape:", lsm_us.shape)
    print("LSM_EU shape:", lsm_eu.shape)

    # ============================================================
    # 8) CV setup: 2-year non-overlapping cyclic blocks
    # ============================================================
    n_folds = 11
    year_start = 2000
    year_end = 2022

    all_year_labels = pd.DatetimeIndex(rgd_time_dd).year.to_numpy()
    years_span = np.unique(all_year_labels)
    years_span = years_span[(years_span >= year_start) & (years_span <= year_end)]

    idx_by_year = {int(y): np.where(all_year_labels == y)[0] for y in years_span}

    years_list = sorted(years_span.tolist())
    if len(years_list) % 2 != 0:
        raise RuntimeError(f"Odd number of available years ({len(years_list)})")

    blocks = [np.array(years_list[i:i + 2], dtype=int) for i in range(0, len(years_list), 2)]
    blocks = blocks[:n_folds]
    n_blocks = len(blocks)

    fold_test_years = []
    fold_val_years = []
    fold_train_years = []

    for i in range(n_blocks):
        test_block = blocks[i]
        val_block = blocks[(i + 1) % n_blocks]
        train_years = np.array(
            sorted(set(years_span.tolist()) - set(test_block.tolist()) - set(val_block.tolist())),
            dtype=int,
        )
        fold_test_years.append(test_block)
        fold_val_years.append(val_block)
        fold_train_years.append(train_years)

    fold_test = [np.sort(np.concatenate([idx_by_year[int(y)] for y in block])) for block in fold_test_years]
    fold_val = [np.sort(np.concatenate([idx_by_year[int(y)] for y in block])) for block in fold_val_years]
    fold_train = [np.sort(np.concatenate([idx_by_year[int(y)] for y in yrs])) for yrs in fold_train_years]

    for i in range(n_blocks):
        print(
            f"Fold {i:02d} | "
            f"test={fold_test_years[i].tolist()} | "
            f"val={fold_val_years[i].tolist()} | "
            f"train={fold_train_years[i].tolist()} | "
            f"n_train_days={len(fold_train[i])} | "
            f"n_val_days={len(fold_val[i])} | "
            f"n_test_days={len(fold_test[i])}"
        )

    # ============================================================
    # Helper functions
    # ============================================================
    def add_hail_stats_rows(rows, fold, dataset_name, months_flat, hb_flat, sz_flat, threshold_cm,
                            ds_ratio=None, ds_div_factor=None):
        for m in range(1, 13):
            mask_m = (months_flat == m)
            if not np.any(mask_m):
                continue
            hb_m = hb_flat[mask_m]
            sz_m = sz_flat[mask_m]
            rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "month": m,
                "large_hail": int(np.sum((hb_m == 1) & (sz_m >= threshold_cm))),
                "no_hail": int(np.sum(hb_m == 0)),
                "ds_ratio": ds_ratio,
                "ds_div_factor": ds_div_factor,
            })

    def add_negative_reduction_rows(rows, fold, dataset_name, months_flat, hb_flat, sz_flat, threshold_cm,
                                    neg_mask_all=None, neg_mask_kept=None, ds_div_factor=None):
        for m in range(1, 13):
            sel_m = (months_flat == m)
            if not np.any(sel_m):
                continue

            hb_m = hb_flat[sel_m]
            sz_m = sz_flat[sel_m]

            n_large = int(np.sum((hb_m == 1) & (sz_m >= threshold_cm)))
            n_nohail_total = int(np.sum(hb_m == 0))
            n_nohail_after_mask = int(np.sum(neg_mask_all[sel_m])) if neg_mask_all is not None else np.nan
            n_nohail_kept_ds = int(np.sum(neg_mask_kept[sel_m])) if neg_mask_kept is not None else np.nan

            rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "month": m,
                "large_hail": n_large,
                "no_hail_total_land": n_nohail_total,
                "no_hail_after_mask": n_nohail_after_mask,
                "no_hail_removed_by_mask": n_nohail_total - n_nohail_after_mask if not pd.isna(n_nohail_after_mask) else np.nan,
                "no_hail_kept_after_ds": n_nohail_kept_ds,
                "no_hail_removed_by_ds": (
                    n_nohail_after_mask - n_nohail_kept_ds
                    if not pd.isna(n_nohail_after_mask) and not pd.isna(n_nohail_kept_ds) else np.nan
                ),
                "ds_div_factor": ds_div_factor,
            })

    def add_daylevel_stats_rows(rows, fold, dataset_name, dates_1d, event_day_1d):
        months = pd.DatetimeIndex(dates_1d).month.to_numpy()
        for m in range(1, 13):
            sel = (months == m)
            if not np.any(sel):
                continue
            hail_days = int(np.sum(event_day_1d[sel]))
            n_days = int(sel.sum())
            rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "month": m,
                "hail_days": hail_days,
                "no_hail_days": n_days - hail_days,
                "n_days_total": n_days,
            })

    def add_shared_cls_stats_rows(rows, fold, dataset_name, dates_1d, cls_1d):
        months = pd.DatetimeIndex(dates_1d).month.to_numpy()
        for m in range(1, 13):
            sel = (months == m)
            if not np.any(sel):
                continue
            cls_m = cls_1d[sel]
            rows.append({
                "fold": fold,
                "dataset": dataset_name,
                "month": m,
                "none_days": int(np.sum(cls_m == 0)),
                "us_only_days": int(np.sum(cls_m == 1)),
                "eu_only_days": int(np.sum(cls_m == 2)),
                "both_days": int(np.sum(cls_m == 3)),
                "any_region_days": int(np.sum(cls_m > 0)),
                "n_days_total": int(sel.sum()),
            })

    def add_merged_hail_stats_rows(rows, fold, dataset_name,
                                   months_flat_us, hb_flat_us, sz_flat_us,
                                   months_flat_eu, hb_flat_eu, sz_flat_eu,
                                   threshold_cm, ds_ratio=None, ds_div_factor=None):
        add_hail_stats_rows(
            rows, fold, dataset_name,
            np.concatenate([months_flat_us, months_flat_eu]),
            np.concatenate([hb_flat_us, hb_flat_eu]),
            np.concatenate([sz_flat_us, sz_flat_eu]),
            threshold_cm, ds_ratio=ds_ratio, ds_div_factor=ds_div_factor
        )

    def prepare_binary_obs(y_hb, y_sz, dates_1d, threshold_cm):
        land_mask = ~np.isnan(y_hb)

        dates_1d = pd.DatetimeIndex(dates_1d)
        month_3d = np.broadcast_to(dates_1d.month.to_numpy()[:, None, None], y_hb.shape)
        year_3d = np.broadcast_to(dates_1d.year.to_numpy()[:, None, None], y_hb.shape)

        month_flat = month_3d[land_mask]
        year_flat = year_3d[land_mask]
        hb_flat = y_hb[land_mask]
        sz_flat = y_sz[land_mask]

        valid_binary = (hb_flat == 0) | ((hb_flat == 1) & (sz_flat >= threshold_cm))

        month_flat_bin = month_flat[valid_binary]
        year_flat_bin = year_flat[valid_binary]
        hb_flat_bin = hb_flat[valid_binary]
        sz_flat_bin = sz_flat[valid_binary]

        y_bin = np.where(
            (hb_flat_bin == 1) & (sz_flat_bin >= threshold_cm), 1, 0
        ).astype(np.int8)

        return land_mask, year_flat_bin, month_flat_bin, hb_flat_bin, sz_flat_bin, valid_binary, y_bin

    def flatten_predictors(X_grid, land_mask, valid_binary):
        X_f = np.moveaxis(X_grid, 1, 0)[:, land_mask]
        X_f[ix_vs03, :] = np.abs(X_f[ix_vs03, :])
        X_f[ix_vs06, :] = np.abs(X_f[ix_vs06, :])
        return X_f[:, valid_binary].T

    def downsample_negatives(y, ds_factor, rng):
        idx_neg = np.where(y == 0)[0]
        idx_pos = np.where(y == 1)[0]

        n_sel = int(len(idx_neg) / ds_factor) if len(idx_neg) > 0 else 0
        sel_neg = rng.choice(idx_neg, size=n_sel, replace=False) if n_sel > 0 else np.array([], dtype=int)
        idx_keep = np.sort(np.concatenate([sel_neg, idx_pos]))
        return idx_keep, idx_neg, sel_neg

    def monthly_clim_freq(months_pos, pseudocount=0.0):
        counts = np.array([np.sum(np.asarray(months_pos) == m) for m in range(1, 13)], dtype=float)
        if pseudocount > 0:
            counts = counts + pseudocount
        freq = counts / counts.sum()
        return {m: float(freq[m - 1]) for m in range(1, 13)}

    def compute_yearmonth_weights_outliers(
        y, years, months, clim_freq,
        low_q=0.1, high_q=0.9,
        w_normal=1.0,
        w_high=0.7,     # too many events → downweight
        w_low=0.5,      # too few events → strong downweight
    ):
        """
        Weight only extreme month-year outliers.
    
        - high event months → slight downweight
        - low / zero months → strong downweight
        - normal months → 1.0
        """
    
        weights = np.ones(len(y), dtype=np.float32)
        pos = (y == 1)
    
        if np.sum(pos) == 0:
            return weights
    
        # --- compute ym fractions ---
        ym_keys = sorted(set(zip(years[pos], months[pos])))
        ym_frac = {}
    
        for yy, mm in ym_keys:
            year_mask = pos & (years == yy)
            year_total = int(np.sum(year_mask))
            if year_total == 0:
                continue
    
            ym_count = int(np.sum(pos & (years == yy) & (months == mm)))
            ym_frac[(yy, mm)] = ym_count / year_total
    
        # --- compute deviations ---
        dev_values = []
        for (yy, mm), frac in ym_frac.items():
            dev = frac - clim_freq[mm]
            dev_values.append(dev)
    
        dev_values = np.array(dev_values)
    
        if len(dev_values) < 5:
            return weights
    
        # --- determine thresholds ---
        low_thr = np.quantile(dev_values, low_q)
        high_thr = np.quantile(dev_values, high_q)
    
        # --- assign weights ---
        for (yy, mm), frac in ym_frac.items():
            dev = frac - clim_freq[mm]
    
            if dev <= low_thr:
                w = w_low   # too few events
            elif dev >= high_thr:
                w = w_high  # too many events
            else:
                w = w_normal
    
            weights[pos & (years == yy) & (months == mm)] = float(w)
    
        return weights

    def compute_threshold_ranking_metrics(y_true, y_prob, thr):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "thr": float(thr),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if (y_pred.sum() not in (0, len(y_pred))) else np.nan,
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "roc-auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        }
    def plot_combined_pr_curves(curves, outpath, title, ylim=(0, 0.1)):
        plt.close("all")
        
        fig, ax = plt.subplots(figsize=(8.2, 5.8))

        auc_values = [c["aucpr"] for c in curves]
        norm = Normalize(vmin=min(auc_values), vmax=max(auc_values))
        cmap = plt.get_cmap("PuOr_r")

        lines = []
        labels = []
        baselines = []

        for c in sorted(curves, key=lambda x: x["aucpr"]):
            line, = ax.plot(c["recall"], c["precision"], linewidth=1.3, alpha=0.95, color=cmap(norm(c["aucpr"])))
            lines.append(line)
            labels.append(f'{c["fold"]:02d}: {c["aucpr"]:.4f}')
            baselines.append(c["baseline"])

        if len(baselines) > 0:
            ax.axhline(np.mean(baselines), linestyle=":", linewidth=1.8, color="black")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        legend = ax.legend(lines, labels, title="Fold : PR-AUC", loc="upper right", fontsize=8, title_fontsize=9)
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", [])
        for h in handles:
            h.set_visible(False)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("PR-AUC")

        plt.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_combined_roc_curves(curves, outpath, title):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(8.2, 5.8))

        auc_values = [c["roc_auc"] for c in curves]
        norm = Normalize(vmin=min(auc_values), vmax=max(auc_values))
        cmap = plt.get_cmap("PuOr_r")

        lines = []
        labels = []

        for c in sorted(curves, key=lambda x: x["roc_auc"]):
            line, = ax.plot(c["fpr"], c["tpr"], linewidth=1.3, alpha=0.95, color=cmap(norm(c["roc_auc"])))
            lines.append(line)
            labels.append(f'{c["fold"]:02d}: {c["roc_auc"]:.4f}')

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="black")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        legend = ax.legend(lines, labels, title="Fold : ROC-AUC", loc="lower right", fontsize=8, title_fontsize=9)
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", [])
        for h in handles:
            h.set_visible(False)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("ROC-AUC")

        plt.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    
    # ============================================================
    # 9) Containers
    # ============================================================
    cv_results = []
    hail_stats_rows = []
    hail_reduction_rows = []
    day_stats_rows = []
    day_cls_stats_rows = []
    pr_curves_valid = []
    pr_curves_valid_untuned = []
    roc_curves_valid = []
    roc_curves_valid_untuned = []

    best_trial_rows = []
    overfit_rows = []


   # ============================================================
    # 10) CV loop
    # ============================================================
    for ii, (tr_idx, val_idx, te_idx) in enumerate(zip(fold_train, fold_val, fold_test)):
        print("\n" + "=" * 70)
        print(
            f"FOLD {ii:02d} | "
            f"train_days={len(tr_idx)} | val_days={len(val_idx)} | test_days={len(te_idx)} | "
            f"thr={threshold_cm:.1f}cm | ds_factor_us={ds_factor_us} | ds_factor_eu={ds_factor_eu}"
        )
        print("=" * 70)

        train_full_idx = np.sort(np.concatenate([tr_idx, val_idx]))

        dates_tr = pd.to_datetime(rgd_time_dd[tr_idx])
        dates_train = pd.to_datetime(rgd_time_dd[train_full_idx])
        dates_val = pd.to_datetime(rgd_time_dd[val_idx])
        dates_te = pd.to_datetime(rgd_time_dd[te_idx])

        # ------------------------------------------------------------
        # Fold-specific monthly climatology from training years only
        # ------------------------------------------------------------
        all_months_pos_us = []
        all_months_pos_eu = []

        for t in tr_idx:
            m = rgd_time_dd[t].month

            hb_us = rgr_noaa_obs[0, t, lsm_us >= 0.5]
            sz_us = rgr_noaa_obs[1, t, lsm_us >= 0.5]
            n_pos_us = int(np.sum((hb_us == 1) & (sz_us >= threshold_cm)))
            all_months_pos_us.extend([m] * n_pos_us)

            hb_eu = rgr_essl_obs[0, t, lsm_eu >= 0.5]
            sz_eu = rgr_essl_obs[1, t, lsm_eu >= 0.5]
            n_pos_eu = int(np.sum((hb_eu == 1) & (sz_eu >= threshold_cm)))
            all_months_pos_eu.extend([m] * n_pos_eu)

        clim_freq_us = monthly_clim_freq(np.array(all_months_pos_us, dtype=int), pseudocount=0.0)
        clim_freq_eu = monthly_clim_freq(np.array(all_months_pos_eu, dtype=int), pseudocount=1.0)

        # ------------------------------------------------------------
        # Observation cubes
        # ------------------------------------------------------------
        hail_bin_us = rgr_noaa_obs[0, :, :, :]
        hail_size_us = rgr_noaa_obs[1, :, :, :]
        hail_bin_eu = rgr_essl_obs[0, :, :, :]
        hail_size_eu = rgr_essl_obs[1, :, :, :]

        y_hb_train_us = np.array(hail_bin_us[train_full_idx, :, :]).astype(float)
        y_sz_train_us = np.array(hail_size_us[train_full_idx, :, :]).astype(float)
        y_hb_tr_us = np.array(hail_bin_us[tr_idx, :, :]).astype(float)
        y_sz_tr_us = np.array(hail_size_us[tr_idx, :, :]).astype(float)
        y_hb_val_us = np.array(hail_bin_us[val_idx, :, :]).astype(float)
        y_sz_val_us = np.array(hail_size_us[val_idx, :, :]).astype(float)
        y_hb_te_us = np.array(hail_bin_us[te_idx, :, :]).astype(float)
        y_sz_te_us = np.array(hail_size_us[te_idx, :, :]).astype(float)

        y_hb_train_eu = np.array(hail_bin_eu[train_full_idx, :, :]).astype(float)
        y_sz_train_eu = np.array(hail_size_eu[train_full_idx, :, :]).astype(float)
        y_hb_tr_eu = np.array(hail_bin_eu[tr_idx, :, :]).astype(float)
        y_sz_tr_eu = np.array(hail_size_eu[tr_idx, :, :]).astype(float)
        y_hb_val_eu = np.array(hail_bin_eu[val_idx, :, :]).astype(float)
        y_sz_val_eu = np.array(hail_size_eu[val_idx, :, :]).astype(float)
        y_hb_te_eu = np.array(hail_bin_eu[te_idx, :, :]).astype(float)
        y_sz_te_eu = np.array(hail_size_eu[te_idx, :, :]).astype(float)

        for arr in (y_hb_train_us, y_sz_train_us, y_hb_tr_us, y_sz_tr_us, y_hb_val_us, y_sz_val_us, y_hb_te_us, y_sz_te_us):
            arr[:, lsm_us < 0.5] = np.nan
        for arr in (y_hb_train_eu, y_sz_train_eu, y_hb_tr_eu, y_sz_tr_eu, y_hb_val_eu, y_sz_val_eu, y_hb_te_eu, y_sz_te_eu):
            arr[:, lsm_eu < 0.5] = np.nan

        # ------------------------------------------------------------
        # Flatten observations
        # ------------------------------------------------------------
        land_mask_tr_us, year_flat_tr_us, month_flat_tr_us, hb_flat_tr_us, sz_flat_tr_us, valid_binary_tr_us, y_tr_us = \
            prepare_binary_obs(y_hb_tr_us, y_sz_tr_us, dates_tr, threshold_cm)
        land_mask_train_us, year_flat_train_us, month_flat_train_us, hb_flat_train_us, sz_flat_train_us, valid_binary_train_us, y_train_us = \
            prepare_binary_obs(y_hb_train_us, y_sz_train_us, dates_train, threshold_cm)
        land_mask_val_us, year_flat_val_us, month_flat_val_us, hb_flat_val_us, sz_flat_val_us, valid_binary_val_us, y_val_us = \
            prepare_binary_obs(y_hb_val_us, y_sz_val_us, dates_val, threshold_cm)
        land_mask_te_us, year_flat_te_us, month_flat_te_us, hb_flat_te_us, sz_flat_te_us, valid_binary_te_us, y_test_us = \
            prepare_binary_obs(y_hb_te_us, y_sz_te_us, dates_te, threshold_cm)

        land_mask_tr_eu, year_flat_tr_eu, month_flat_tr_eu, hb_flat_tr_eu, sz_flat_tr_eu, valid_binary_tr_eu, y_tr_eu = \
            prepare_binary_obs(y_hb_tr_eu, y_sz_tr_eu, dates_tr, threshold_cm)
        land_mask_train_eu, year_flat_train_eu, month_flat_train_eu, hb_flat_train_eu, sz_flat_train_eu, valid_binary_train_eu, y_train_eu = \
            prepare_binary_obs(y_hb_train_eu, y_sz_train_eu, dates_train, threshold_cm)
        land_mask_val_eu, year_flat_val_eu, month_flat_val_eu, hb_flat_val_eu, sz_flat_val_eu, valid_binary_val_eu, y_val_eu = \
            prepare_binary_obs(y_hb_val_eu, y_sz_val_eu, dates_val, threshold_cm)
        land_mask_te_eu, year_flat_te_eu, month_flat_te_eu, hb_flat_te_eu, sz_flat_te_eu, valid_binary_te_eu, y_test_eu = \
            prepare_binary_obs(y_hb_te_eu, y_sz_te_eu, dates_te, threshold_cm)

        # Stats
        add_hail_stats_rows(hail_stats_rows, ii, "US_tr_noDS", month_flat_tr_us, hb_flat_tr_us, sz_flat_tr_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_train_noDS", month_flat_train_us, hb_flat_train_us, sz_flat_train_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_val_noDS", month_flat_val_us, hb_flat_val_us, sz_flat_val_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_test_noDS", month_flat_te_us, hb_flat_te_us, sz_flat_te_us, threshold_cm)

        add_hail_stats_rows(hail_stats_rows, ii, "EU_tr_noDS", month_flat_tr_eu, hb_flat_tr_eu, sz_flat_tr_eu, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "EU_train_noDS", month_flat_train_eu, hb_flat_train_eu, sz_flat_train_eu, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "EU_val_noDS", month_flat_val_eu, hb_flat_val_eu, sz_flat_val_eu, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "EU_test_noDS", month_flat_te_eu, hb_flat_te_eu, sz_flat_te_eu, threshold_cm)

        # Day-level stats
        event_day_tr_us = np.any((y_hb_tr_us == 1) & (y_sz_tr_us >= threshold_cm), axis=(1, 2))
        event_day_tr_eu = np.any((y_hb_tr_eu == 1) & (y_sz_tr_eu >= threshold_cm), axis=(1, 2))
        cls_tr = event_day_tr_us.astype(np.int8) + 2 * event_day_tr_eu.astype(np.int8)

        event_day_val_us = np.any((y_hb_val_us == 1) & (y_sz_val_us >= threshold_cm), axis=(1, 2))
        event_day_val_eu = np.any((y_hb_val_eu == 1) & (y_sz_val_eu >= threshold_cm), axis=(1, 2))
        cls_val = event_day_val_us.astype(np.int8) + 2 * event_day_val_eu.astype(np.int8)

        add_daylevel_stats_rows(day_stats_rows, ii, "US_tr_days", dates_tr, event_day_tr_us)
        add_daylevel_stats_rows(day_stats_rows, ii, "EU_tr_days", dates_tr, event_day_tr_eu)
        add_daylevel_stats_rows(day_stats_rows, ii, "tr_days_anyRegion", dates_tr, (event_day_tr_us | event_day_tr_eu))
        add_shared_cls_stats_rows(day_cls_stats_rows, ii, "tr_days_anyRegion_cls", dates_tr, cls_tr)

        add_daylevel_stats_rows(day_stats_rows, ii, "US_val_days", dates_val, event_day_val_us)
        add_daylevel_stats_rows(day_stats_rows, ii, "EU_val_days", dates_val, event_day_val_eu)
        add_daylevel_stats_rows(day_stats_rows, ii, "val_days_anyRegion", dates_val, (event_day_val_us | event_day_val_eu))
        add_shared_cls_stats_rows(day_cls_stats_rows, ii, "val_days_anyRegion_cls", dates_val, cls_val)

        # ------------------------------------------------------------
        # Negative downsampling
        # ------------------------------------------------------------
        idx_keep_tr_us, idx_no_us_raw, sel_no_us = downsample_negatives(y_tr_us, ds_factor_us, np.random.default_rng(123 + ii))
        idx_keep_train_us, idx_no_train_us_raw, sel_no_train_us = downsample_negatives(y_train_us, ds_factor_us, np.random.default_rng(223 + ii))

        idx_keep_tr_eu, idx_no_eu_raw, sel_no_eu = downsample_negatives(y_tr_eu, ds_factor_eu, np.random.default_rng(456 + ii))
        idx_keep_train_eu, idx_no_train_eu_raw, sel_no_train_eu = downsample_negatives(y_train_eu, ds_factor_eu, np.random.default_rng(556 + ii))

        y_tr_bal_us = y_tr_us[idx_keep_tr_us]
        y_train_bal_us = y_train_us[idx_keep_train_us]
        y_tr_bal_eu = y_tr_eu[idx_keep_tr_eu]
        y_train_bal_eu = y_train_eu[idx_keep_train_eu]

        neg_mask_all_us = np.zeros_like(hb_flat_tr_us, dtype=bool)
        neg_mask_all_us[idx_no_us_raw] = True
        neg_mask_kept_us = np.zeros_like(hb_flat_tr_us, dtype=bool)
        neg_mask_kept_us[sel_no_us] = True

        neg_mask_all_train_us = np.zeros_like(hb_flat_train_us, dtype=bool)
        neg_mask_all_train_us[idx_no_train_us_raw] = True
        neg_mask_kept_train_us = np.zeros_like(hb_flat_train_us, dtype=bool)
        neg_mask_kept_train_us[sel_no_train_us] = True

        neg_mask_all_eu = np.zeros_like(hb_flat_tr_eu, dtype=bool)
        neg_mask_all_eu[idx_no_eu_raw] = True
        neg_mask_kept_eu = np.zeros_like(hb_flat_tr_eu, dtype=bool)
        neg_mask_kept_eu[sel_no_eu] = True

        neg_mask_all_train_eu = np.zeros_like(hb_flat_train_eu, dtype=bool)
        neg_mask_all_train_eu[idx_no_train_eu_raw] = True
        neg_mask_kept_train_eu = np.zeros_like(hb_flat_train_eu, dtype=bool)
        neg_mask_kept_train_eu[sel_no_train_eu] = True

        add_negative_reduction_rows(
            hail_reduction_rows, ii, "US_tr_DS_reduction",
            month_flat_tr_us, hb_flat_tr_us, sz_flat_tr_us, threshold_cm,
            neg_mask_all_us, neg_mask_kept_us,
            ds_div_factor=(len(idx_no_us_raw) / max(1, len(sel_no_us))),
        )
        add_negative_reduction_rows(
            hail_reduction_rows, ii, "US_train_DS_reduction",
            month_flat_train_us, hb_flat_train_us, sz_flat_train_us, threshold_cm,
            neg_mask_all_train_us, neg_mask_kept_train_us,
            ds_div_factor=(len(idx_no_train_us_raw) / max(1, len(sel_no_train_us))),
        )
        add_negative_reduction_rows(
            hail_reduction_rows, ii, "EU_tr_DS_reduction",
            month_flat_tr_eu, hb_flat_tr_eu, sz_flat_tr_eu, threshold_cm,
            neg_mask_all_eu, neg_mask_kept_eu,
            ds_div_factor=(len(idx_no_eu_raw) / max(1, len(sel_no_eu))),
        )
        add_negative_reduction_rows(
            hail_reduction_rows, ii, "EU_train_DS_reduction",
            month_flat_train_eu, hb_flat_train_eu, sz_flat_train_eu, threshold_cm,
            neg_mask_all_train_eu, neg_mask_kept_train_eu,
            ds_div_factor=(len(idx_no_train_eu_raw) / max(1, len(sel_no_train_eu))),
        )

        # ------------------------------------------------------------
        # Predictor flattening
        # ------------------------------------------------------------
        X_grid_tr_us = np.copy(rgr_era_varall_us[tr_idx, :, :, :]).astype(np.float32)
        X_grid_train_us = np.copy(rgr_era_varall_us[train_full_idx, :, :, :]).astype(np.float32)
        X_grid_val_us = np.copy(rgr_era_varall_us[val_idx, :, :, :]).astype(np.float32)
        X_grid_te_us = np.copy(rgr_era_varall_us[te_idx, :, :, :]).astype(np.float32)

        X_grid_tr_eu = np.copy(rgr_era_varall_eu[tr_idx, :, :, :]).astype(np.float32)
        X_grid_train_eu = np.copy(rgr_era_varall_eu[train_full_idx, :, :, :]).astype(np.float32)
        X_grid_val_eu = np.copy(rgr_era_varall_eu[val_idx, :, :, :]).astype(np.float32)
        X_grid_te_eu = np.copy(rgr_era_varall_eu[te_idx, :, :, :]).astype(np.float32)

        X_grid_tr_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_train_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_val_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_te_us[:, :, lsm_us < 0.5] = np.nan

        X_grid_tr_eu[:, :, lsm_eu < 0.5] = np.nan
        X_grid_train_eu[:, :, lsm_eu < 0.5] = np.nan
        X_grid_val_eu[:, :, lsm_eu < 0.5] = np.nan
        X_grid_te_eu[:, :, lsm_eu < 0.5] = np.nan

        X_tr_us_bin = flatten_predictors(X_grid_tr_us, land_mask_tr_us, valid_binary_tr_us)
        X_train_us_bin = flatten_predictors(X_grid_train_us, land_mask_train_us, valid_binary_train_us)
        X_val_us_bin = flatten_predictors(X_grid_val_us, land_mask_val_us, valid_binary_val_us)
        X_te_us_bin = flatten_predictors(X_grid_te_us, land_mask_te_us, valid_binary_te_us)

        X_tr_eu_bin = flatten_predictors(X_grid_tr_eu, land_mask_tr_eu, valid_binary_tr_eu)
        X_train_eu_bin = flatten_predictors(X_grid_train_eu, land_mask_train_eu, valid_binary_train_eu)
        X_val_eu_bin = flatten_predictors(X_grid_val_eu, land_mask_val_eu, valid_binary_val_eu)
        X_te_eu_bin = flatten_predictors(X_grid_te_eu, land_mask_te_eu, valid_binary_te_eu)

        X_tr_bal_us = X_tr_us_bin[idx_keep_tr_us, :]
        X_train_bal_us = X_train_us_bin[idx_keep_train_us, :]
        X_tr_bal_eu = X_tr_eu_bin[idx_keep_tr_eu, :]
        X_train_bal_eu = X_train_eu_bin[idx_keep_train_eu, :]

        month_tr_bal_us = month_flat_tr_us[idx_keep_tr_us]
        year_tr_bal_us = year_flat_tr_us[idx_keep_tr_us]
        month_train_bal_us = month_flat_train_us[idx_keep_train_us]
        year_train_bal_us = year_flat_train_us[idx_keep_train_us]

        month_tr_bal_eu = month_flat_tr_eu[idx_keep_tr_eu]
        year_tr_bal_eu = year_flat_tr_eu[idx_keep_tr_eu]
        month_train_bal_eu = month_flat_train_eu[idx_keep_train_eu]
        year_train_bal_eu = year_flat_train_eu[idx_keep_train_eu]

        # ------------------------------------------------------------
        # Dataset merges
        # ------------------------------------------------------------
        X_tr_bal = np.concatenate((X_tr_bal_us, X_tr_bal_eu), axis=0)
        y_tr_bal = np.concatenate((y_tr_bal_us, y_tr_bal_eu), axis=0)

        X_train_bal = np.concatenate((X_train_bal_us, X_train_bal_eu), axis=0)
        y_train_bal = np.concatenate((y_train_bal_us, y_train_bal_eu), axis=0)

        X_tr = np.concatenate((X_tr_us_bin, X_tr_eu_bin), axis=0)
        y_tr = np.concatenate((y_tr_us, y_tr_eu), axis=0)

        X_val = np.concatenate((X_val_us_bin, X_val_eu_bin), axis=0)
        y_val = np.concatenate((y_val_us, y_val_eu), axis=0)

        X_test = np.concatenate((X_te_us_bin, X_te_eu_bin), axis=0)
        y_test = np.concatenate((y_test_us, y_test_eu), axis=0)

        # ------------------------------------------------------------
        # Year-month weights for positives only
        # ------------------------------------------------------------
        # sw_tr_us = compute_yearmonth_weights_outliers(
        #     y_tr_bal_us, year_tr_bal_us, month_tr_bal_us, clim_freq_us,
        # )
        # sw_train_us = compute_yearmonth_weights_outliers(
        #     y_train_bal_us, year_train_bal_us, month_train_bal_us, clim_freq_us,
        # )

        # sw_tr_eu = compute_yearmonth_weights_outliers(
        #     y_tr_bal_eu, year_tr_bal_eu, month_tr_bal_eu, clim_freq_eu,
        # )
        # sw_train_eu = compute_yearmonth_weights_outliers(
        #     y_train_bal_eu, year_train_bal_eu, month_train_bal_eu, clim_freq_eu,
        # )

        # sw_tr_us *= target_us_share
        # sw_train_us *= target_us_share
        # sw_tr_eu *= target_eu_share
        # sw_train_eu *= target_eu_share

        # sw_tr = np.concatenate((sw_tr_us, sw_tr_eu), axis=0)
        # sw_train = np.concatenate((sw_train_us, sw_train_eu), axis=0)

        # sw_tr = sw_tr / np.mean(sw_tr)
        # sw_train = sw_train / np.mean(sw_train)

        # print(f"Fold {ii:02d} | sample weights: mean={np.mean(sw_tr):.4f}, min={np.min(sw_tr):.4f}, max={np.max(sw_tr):.4f}")

        # ------------------------------------------------------------
        # More merged stats
        # ------------------------------------------------------------
        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_tr_noDS",
            month_flat_tr_us, hb_flat_tr_us, sz_flat_tr_us,
            month_flat_tr_eu, hb_flat_tr_eu, sz_flat_tr_eu,
            threshold_cm,
        )

        ds_ratio_tr_any = ((np.sum(y_tr_bal_us == 0) + np.sum(y_tr_bal_eu == 0)) /
                           max(1, (np.sum(y_tr_bal_us == 1) + np.sum(y_tr_bal_eu == 1))))

        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_tr_DS",
            month_flat_tr_us[idx_keep_tr_us], hb_flat_tr_us[idx_keep_tr_us], sz_flat_tr_us[idx_keep_tr_us],
            month_flat_tr_eu[idx_keep_tr_eu], hb_flat_tr_eu[idx_keep_tr_eu], sz_flat_tr_eu[idx_keep_tr_eu],
            threshold_cm, ds_ratio=ds_ratio_tr_any
        )

        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_train_noDS",
            month_flat_train_us, hb_flat_train_us, sz_flat_train_us,
            month_flat_train_eu, hb_flat_train_eu, sz_flat_train_eu,
            threshold_cm,
        )

        ds_ratio_train_any = ((np.sum(y_train_bal_us == 0) + np.sum(y_train_bal_eu == 0)) /
                              max(1, (np.sum(y_train_bal_us == 1) + np.sum(y_train_bal_eu == 1))))

        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_train_DS",
            month_flat_train_us[idx_keep_train_us], hb_flat_train_us[idx_keep_train_us], sz_flat_train_us[idx_keep_train_us],
            month_flat_train_eu[idx_keep_train_eu], hb_flat_train_eu[idx_keep_train_eu], sz_flat_train_eu[idx_keep_train_eu],
            threshold_cm, ds_ratio=ds_ratio_train_any
        )

        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_val_noDS",
            month_flat_val_us, hb_flat_val_us, sz_flat_val_us,
            month_flat_val_eu, hb_flat_val_eu, sz_flat_val_eu,
            threshold_cm,
        )

        add_merged_hail_stats_rows(
            hail_stats_rows, ii, "ANY_test_noDS",
            month_flat_te_us, hb_flat_te_us, sz_flat_te_us,
            month_flat_te_eu, hb_flat_te_eu, sz_flat_te_eu,
            threshold_cm,
        )

        # ------------------------------------------------------------
        # DMatrices without sample weights
        # ------------------------------------------------------------
        dtr_balanced = xgb.DMatrix(
            X_tr_bal, label=y_tr_bal,
            missing=np.nan, feature_names=feature_names
        )
        dtrain_balanced = xgb.DMatrix(
            X_train_bal, label=y_train_bal,
            missing=np.nan, feature_names=feature_names
        )
        dtr_imbalanced = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_names)
        dval_imbalanced = xgb.DMatrix(X_val, label=y_val, missing=np.nan, feature_names=feature_names)
        dte = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=feature_names)

        # ------------------------------------------------------------
        # Baseline model
        # ------------------------------------------------------------
        global_prevalence = float((y_train_us.sum() + y_train_eu.sum()) / (len(y_train_us) + len(y_train_eu)))
        print(f"Global prevalence (US+EU, Train years): {global_prevalence:.6f}")

        base_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "device": "cuda",
            "max_bin": 256,
            "seed": 42 + ii,
            "base_score": global_prevalence,
        }

        baseline_trial_params = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "alpha": 1.0,
            "lambda": 1.0,
            "gamma": 0.0,
        }

        baseline_params = {**base_params, **baseline_trial_params}

        evals_result = {}
        baseline_model = xgb.train(
            params=baseline_params,
            dtrain=dtr_balanced,
            num_boost_round=200,
            evals=[(dtr_balanced, "train"), (dval_imbalanced, "valid")],
            early_stopping_rounds=40,
            evals_result=evals_result,
            verbose_eval=False,
        )

        prevalence_valid = np.mean(y_val)
        y_val_pred_untuned = baseline_model.predict(dval_imbalanced)
        sklearn_ap_valid_untuned = average_precision_score(y_val, y_val_pred_untuned)
        xgb_aucpr_valid_untuned = float(baseline_model.best_score)
        precision_valid_untuned, recall_valid_untuned, _= precision_recall_curve(y_val, y_val_pred_untuned)
        aucpr_valid_untuned = auc(recall_valid_untuned, precision_valid_untuned)
        fpr_valid_untuned, tpr_valid_untuned, _= roc_curve(y_val, y_val_pred_untuned)
        aucroc_valid_untuned = roc_auc_score(y_val, y_val_pred_untuned)

        pr_curves_valid_untuned.append({
            "fold": ii,
            "recall": recall_valid_untuned,
            "precision": precision_valid_untuned,
            "aucpr": aucpr_valid_untuned,
            "baseline": prevalence_valid,
        })

        roc_curves_valid_untuned.append({
             "fold": ii,
             "fpr": fpr_valid_untuned,
             "tpr": tpr_valid_untuned,
             "roc_auc": aucroc_valid_untuned,
        })

        print(
            f"[BASELINE] fold={ii:02d} | "
            f"best_iter={baseline_model.best_iteration} | "
            f"xgb_aucpr={xgb_aucpr_valid_untuned:.10f} | "
            f"sklearn_valid_ap={sklearn_ap_valid_untuned:.10f}"
        )

        # ------------------------------------------------------------
        # Optuna tuning
        # ------------------------------------------------------------
        base_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "device": "cuda",
            "max_bin": 256,
            "seed": 42 + ii,
            "base_score": global_prevalence,
            # "learning_rate": 0.018,
            # "max_depth":  13,
            # "min_child_weight": 35,
            # "subsample": 0.68,
            # "colsample_bytree": 0.7,
            # "gamma": 0,
            # "alpha": 0.1,
        }

        def objective(trial):
            trial_params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.025, log=True),
                "max_depth": trial.suggest_int("max_depth", 11, 17),
                "min_child_weight": trial.suggest_float("min_child_weight", 28, 42, log=True),
                "subsample": trial.suggest_float("subsample", 0.62, 0.78),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.8),
                "gamma": trial.suggest_float("gamma", 1e-8, 1e-3, log=True),
                "lambda": trial.suggest_float("lambda", 0.06, 0.11),
                "alpha": trial.suggest_float("alpha", 1e-5, 0.3, log=True),
            }
            params = {**base_params, **trial_params}

            pruning_cb = XGBoostPruningCallback(trial, "valid-aucpr")

            evals_result = {}
            booster_hp_tuning = xgb.train(
                params=params,
                dtrain=dtr_balanced,
                num_boost_round=200,
                evals=[(dtr_balanced, "train"), (dval_imbalanced, "valid")],
                early_stopping_rounds=40,
                callbacks=[pruning_cb],
                evals_result=evals_result,
                verbose_eval=False,
            )

            y_val_pred = booster_hp_tuning.predict(dval_imbalanced)
            sklearn_ap_valid_tuned = average_precision_score(y_val, y_val_pred)
            xgb_aucpr_valid_tuned = float(booster_hp_tuning.best_score)
                    
            trial.set_user_attr("best_iteration", int(booster_hp_tuning.best_iteration))
            trial.set_user_attr("xgb_aucpr_valid_tuned", xgb_aucpr_valid_tuned)
            trial.set_user_attr("sklearn_ap_valid_tuned", sklearn_ap_valid_tuned)
            trial.set_user_attr("evals_result", evals_result)

            print(
                f"[TRIAL {trial.number:03d}] "
                f"best_iter={booster_hp_tuning.best_iteration} | "
                f"XGB AUCPR={xgb_aucpr_valid_tuned:.10f} | "
                f"Sklearn AP={sklearn_ap_valid_tuned:.10f} | "
            )

            return float(sklearn_ap_valid_tuned)

        sampler = optuna.samplers.TPESampler(seed=42 + ii)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=15,
            max_resource=200,
            reduction_factor=3,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            objective,
            n_trials=30,
            n_jobs=1,
            show_progress_bar=False,
            gc_after_trial=True,
        )

        best_trial = study.best_trial
        best_iter = best_trial.user_attrs.get("best_iteration")
        best_params = {**base_params, **best_trial.params}
        best_ap_valid = float(best_trial.value)
        best_xgb_aucpr_valid = float(best_trial.user_attrs["xgb_aucpr_valid_tuned"])

        print(
            f"[OPTUNA BEST] fold={ii:02d} | "
            f"best_iter={best_iter}"
            f"valid_XGB_AUCPR={best_xgb_aucpr_valid:.10f} | "
            f"valid_AP={best_ap_valid:.10f} | "
        )
        print("Best params:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        df_complete = (
            df_trials[df_trials["state"].astype(str).str.contains("COMPLETE", regex=False)]
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )

        df_best = df_complete.head(1).copy()
        df_best.insert(0, "rank", 1)
        df_best.insert(0, "fold", ii)
        best_trial_rows.append(df_best)

        print(f"\n=== Best completed trial for fold {ii:02d} saved ===")

        best_csv_path = os.path.join(stats_dir, f"optuna_best_fold{ii}_v2.csv")
        df_best.to_csv(best_csv_path, index=False)
        print(f"Saved best trial to: {best_csv_path}")

        all_csv_path = os.path.join(stats_dir, f"optuna_trials_fold{ii}_v2.csv")
        df_trials.to_csv(all_csv_path, index=False)
        print(f"Saved all Optuna trials to: {all_csv_path}")

        # ------------------------------------------------------------
        # AUCPR learning curve
        # ------------------------------------------------------------
        curves = best_trial.user_attrs.get("evals_result")
        valid_aucpr = curves.get("valid", {}).get("aucpr", [])
        epochs = range(len(valid_aucpr))

        fig = plt.figure(figsize=(6, 4))
        plt.plot(
            epochs,
            valid_aucpr,
            linewidth=0.8,
            color="#dda15e",
            label="Valid. Data",
        )

        plt.axhline(
            prevalence_valid,
            color="#dda15e",
            ls="--",
            lw=1,
            label=f"Validation Baseline (prevalence={prevalence_valid:.4f})"
        )

        plt.ylim(0, 0.02)
        plt.legend()
        plt.xlabel("Number of Boosting Rounds")
        plt.ylabel("AUCPR")
        plt.title("Learning Curve of AUCPR Across Boosting Rounds")

        aucpr_plot_path = os.path.join(plots_dir, f"aucpr_fold{ii}.png")
        fig.savefig(aucpr_plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved AUCPR plot for fold {ii} to: {aucpr_plot_path}")
        
        # ------------------------------------------------------------
        # Final model on train + val
        # ------------------------------------------------------------
        model_diagnostic = xgb.train(
            params=best_params,
            dtrain=dtr_balanced,
            num_boost_round=int(best_iter) + 1,
            verbose_eval=False,
        )

        model_final = xgb.train(
            params=best_params,
            dtrain=dtrain_balanced,
            num_boost_round=int(best_iter)+1,
            verbose_eval=False,
        )

        # ------------------------------------------------------------
        # Predictions
        # ------------------------------------------------------------
        # prediction on valid data
        baseline_valid = float(np.mean(y_val))
        y_val_pred = model_diagnostic.predict(dval_imbalanced)
        precision_valid, recall_valid, _ = precision_recall_curve(y_val, y_val_pred)
        aucpr_valid = auc(recall_valid, precision_valid)
        sklearn_ap_valid_tuned = average_precision_score(y_val, y_val_pred)
        fpr_valid, tpr_valid, _= roc_curve(y_val, y_val_pred)
        aucroc_valid = roc_auc_score(y_val, y_val_pred)

        # prediction on train data
        baseline_train = float(np.mean(y_tr))
        y_train_pred = model_diagnostic.predict(dtr_imbalanced)
        precision_train, recall_train, _ = precision_recall_curve(y_tr, y_train_pred)
        aucpr_train = auc(recall_train, precision_train)
        sklearn_ap_train = average_precision_score(y_tr, y_train_pred)

        # final test prediction
        baseline_test = float(np.mean(y_test))
        y_test_pred = model_final.predict(dte)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
        aucpr_test = auc(recall_test, precision_test)
        sklearn_ap_test = average_precision_score(y_test, y_test_pred)

        ap_gap_train_minus_valid = sklearn_ap_train - sklearn_ap_valid_tuned
        aucpr_gap_train_minus_valid = aucpr_train - aucpr_valid
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(
            recall_train, precision_train,
            linewidth=0.8,
            color="#588157",
            label=f"Train PR curve (AUC={aucpr_train:.4f})"
        )
        ax.plot(
            recall_valid, precision_valid,
            linewidth=0.8,
            color="#dda15e",
            label=f"Validation PR curve (AUC={aucpr_valid:.4f})"
        )

        ax.axhline(
            baseline_valid,
            linestyle=":",
            linewidth=2,
            color="#dda15e",
            label=f"Validation Baseline (prevalence={baseline_valid:.4f})"
        )
        ax.axhline(
            baseline_train,
            linestyle=":",
            linewidth=2,
            color="#588157",
            label=f"Train Baseline (prevalence={baseline_train:.4f})"
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 0.2)
        ax.set_title(f"Precision-Recall Curve (Fold {ii:02d})")
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_locator(MultipleLocator(0.025))
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        pr_curve_path = os.path.join(plots_dir, f"pr_curve_fold{ii:02d}.png")
        fig.savefig(pr_curve_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        # ------------------------------------------------------------
        pr_curves_valid.append({
            "fold": ii,
            "recall": recall_valid,
            "precision": precision_valid,
            "aucpr": aucpr_valid,
            "baseline": baseline_valid,
        })
            
        roc_curves_valid.append({
            "fold": ii,
            "fpr": fpr_valid,
            "tpr": tpr_valid,
            "roc_auc": aucroc_valid,
        })
        # ------------------------------------------------------------
        thr_metrics_test = compute_threshold_ranking_metrics(y_test, y_test_pred, thr=0.48)

        overfit_rows.append(
            {
                "fold": ii,
                "train_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[tr_idx]).year))),
                "val_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[val_idx]).year))),
                "n_train": int(len(y_tr)),
                "n_valid": int(len(y_val)),
                "baseline_train": float(baseline_train),
                "baseline_valid": float(baseline_valid),
                "ap_train": float(sklearn_ap_train),
                "ap_valid": float(sklearn_ap_valid_tuned),
                "ap_gap_train_minus_valid": float(ap_gap_train_minus_valid),
                "aucpr_train": float(aucpr_train),
                "aucpr_valid": float(aucpr_valid),
                "aucpr_gap_train_minus_valid": float(aucpr_gap_train_minus_valid),
            }
        )

        
        cv_results.append(
            {
                "fold": ii,
                "train_years_full": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[train_full_idx]).year))),
                "train_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[tr_idx]).year))),
                "val_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[val_idx]).year))),
                "test_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[te_idx]).year))),
                "sklearn_ap_valid_untuned": float(sklearn_ap_valid_untuned),
                "prevalence_valid": float(prevalence_valid),
                "sklearn_ap_valid_tuned": float(best_ap_valid),
                "xgb_aucpr_valid_tuned": float(best_xgb_aucpr_valid),
                "best_iter": int(best_iter),
                **{f"P_{k}": float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
                "sklearn_ap_test": float(sklearn_ap_test),
                "baseline_test": baseline_test,
                "baseline_train": baseline_train,
                "tp": thr_metrics_test["tp"],
                "tn": thr_metrics_test["tn"],
                "fp": thr_metrics_test["fp"],
                "fn": thr_metrics_test["fn"],
                "n_pos_test": int(np.sum(y_test == 1)),
                "n_neg_test": int(np.sum(y_test == 0)),
                "acc_thr": thr_metrics_test["accuracy"],
                "prec_thr": thr_metrics_test["precision"],
                "rec_thr": thr_metrics_test["recall"],
                "f1_thr": thr_metrics_test["f1"],
                "mcc_thr": thr_metrics_test["mcc"],
            }
        )

        model_path = os.path.join(models_dir, f"xgbGlobal_sigHail_fold{ii}.json")
        model_final.save_model(model_path)
        np.save(os.path.join(preds_dir, f"fold{ii}_y_test_prob.npy"), y_test_pred)

    # ============================================================
    # Combined plots
    # ============================================================
    plot_combined_pr_curves(
        curves=pr_curves_valid,
        outpath=os.path.join(plots_dir, "pr_curves_all_folds_valid.png"),
        title="Precision-Recall Curves on Validation Data (all CV folds)",
        ylim=(0, 0.1),
    )

    plot_combined_roc_curves(
        curves=roc_curves_valid,
        outpath=os.path.join(plots_dir, "roc_curves_all_folds_valid.png"),
        title="ROC Curves on Validation Data (all CV folds)",
    )

    plot_combined_pr_curves(
        curves=pr_curves_valid_untuned,
        outpath=os.path.join(plots_dir, "pr_curves_all_folds_valid_untuned.png"),
        title="Precision-Recall Curves on Validation Data (all CV folds), not tuned",
        ylim=(0, 0.1),
    )

    plot_combined_roc_curves(
        curves=roc_curves_valid_untuned,
        outpath=os.path.join(plots_dir, "roc_curves_all_folds_valid_untuned.png"),
        title="ROC Curves on Validation Data (all CV folds), not tuned",
    )
    
    # ============================================================
    # Save overall results
    # ============================================================
    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(os.path.join(stats_dir, "cv_results_all_folds.csv"), index=False)
    
    print(
        f"[CV] AUCPR TEST max={df_cv['sklearn_ap_test'].max():.6f} | "
        f"median={df_cv['sklearn_ap_test'].median():.6f} | "
        f"mean={df_cv['sklearn_ap_test'].mean():.6f} | "
        f"std={df_cv['sklearn_ap_test'].std():.6f}"
    )

    if len(best_trial_rows) > 0:
        df_best_all = pd.concat(best_trial_rows, ignore_index=True)
        df_best_all = df_best_all.sort_values(["fold", "rank"]).reset_index(drop=True)
        cols_drop = ["user_attrs_evals_result"]
        df_best_excel = df_best_all.drop(columns=[c for c in cols_drop if c in df_best_all.columns])
        
        best_xlsx_path = os.path.join(stats_dir, "optuna_best_all_folds.xlsx")
        df_best_excel.to_excel(best_xlsx_path, index=False, engine="openpyxl")
        print(f"[SAVE] Combined best-trial Excel written to: {best_xlsx_path}")


    if len(overfit_rows) > 0:
        df_overfit = pd.DataFrame(overfit_rows)
        df_overfit.to_csv(os.path.join(stats_dir, "train_valid_gap_by_fold.csv"), index=False)

        print(
            f"[GAP] AP gap train-valid | "
            f"min={df_overfit['ap_gap_train_minus_valid'].min():.6f} | "
            f"median={df_overfit['ap_gap_train_minus_valid'].median():.6f} | "
            f"mean={df_overfit['ap_gap_train_minus_valid'].mean():.6f} | "
            f"max={df_overfit['ap_gap_train_minus_valid'].max():.6f}"
        )

    pd.DataFrame(hail_stats_rows).to_csv(os.path.join(stats_dir, "hail_gridcell_days.csv"), index=False)
    pd.DataFrame(hail_reduction_rows).to_csv(os.path.join(stats_dir, "hail_gridcell_days_reduction.csv"), index=False)
    pd.DataFrame(day_stats_rows).to_csv(os.path.join(stats_dir, "hail_days_train_val.csv"), index=False)
    pd.DataFrame(day_cls_stats_rows).to_csv(os.path.join(stats_dir, "hail_days_shared_cls_train_val.csv"), index=False)

    print(f"[SAVE] Saved {len(df_cv)} fold models and prediction files.")
    print(f"[SAVE] CV results written to: {os.path.join(stats_dir, 'cv_results_all_folds.csv')}")

if __name__ == "__main__":
    main()