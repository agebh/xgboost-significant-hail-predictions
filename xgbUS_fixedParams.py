#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
import traceback
from pathlib import Path
from datetime import datetime

import psutil
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from netCDF4 import Dataset

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# Thread settings
# ============================================================
gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
threads_per_trial = 1

print("gpus:", gpus, "threads_per_trial:", threads_per_trial)
print("SLURM_MEM_PER_NODE:", os.environ.get("SLURM_MEM_PER_NODE"))
print("SLURM_MEM_PER_CPU :", os.environ.get("SLURM_MEM_PER_CPU"))
print("RAM total (GB):", psutil.virtual_memory().total / 1e9)

os.environ["OMP_NUM_THREADS"] = str(threads_per_trial)
os.environ["MKL_NUM_THREADS"] = str(threads_per_trial)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_trial)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_trial)


def main():
    # ============================================================
    # 1) Settings
    # ============================================================
    start_day = datetime(2000, 1, 1, 0)
    stop_day = datetime(2022, 12, 31, 23)
    missing_year = 2003

    threshold_cm = 4.4
    ds_factor_us = 39  
    target_us_share = 1.0

    run_date = "xgbUS_sigHail/final_dall"

    # Paths
    noaa_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/hail_observations/SPC_data_griddedERA")

    era_conus_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/e5_hailpredictors_conus")
    era_invar_dir = Path("/nfs/cumulus/highres_nobackup/agebhardt/e5_data_processing/e5_invariant")

    era_const_fields = era_invar_dir / "e5_invariant_129_z_ll025sc.2020010100_2020010100.nc"
    lsm_file = era_invar_dir / "e5_lsm_11024sc.nc"

    # Bounding boxes
    us_lat_min, us_lat_max = 25.25, 49.0
    us_lon_min, us_lon_max = -130.0, -65.0

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
    # 5) Predictor grids US
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

    # ============================================================
    # 6) Land-sea mask
    # ============================================================
    with Dataset(lsm_file, mode="r") as nc:
        lsm = np.squeeze(nc.variables["lsm"][:]).astype(np.float32)

    idx_lat_glob_us = np.where((rgr_lat >= us_lat_min) & (rgr_lat <= us_lat_max))[0]
    idx_lon_glob_us = np.where((rgr_lon >= us_lon_min) & (rgr_lon <= us_lon_max))[0]
    lsm_us = lsm[idx_lat_glob_us][:, idx_lon_glob_us]

    print("LSM_US shape:", lsm_us.shape)

    # ============================================================
    # 7) CV setup: 2-year non-overlapping cyclic blocks
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

    def boost_months_for_positives(weights, y, months, boost_months=(5, 6, 7), boost_factor=1.2):
        weights = weights.copy()
        mask = (y == 1) & np.isin(months, boost_months)
        weights[mask] *= boost_factor
        return weights

    def compute_targeted_weights_us(y, months):
        weights = np.ones(len(y), dtype=np.float32)
    
        pos = (y == 1)
    
        weights[pos & (months == 4)] *= 0.5   # penalize April
        weights[pos & (months == 5)] *= 1.2   # favor May
        weights[pos & (months == 6)] *= 1.6   # optional
        weights[pos & (months == 7)] *= 1.2   # optional
    
        return weights

    def downsample_positive_months(y, month_flat, month_keep_frac, rng):
        """
        Downsample positive class (y==1) only in selected months.
    
        Parameters
        ----------
        y : 1D array
            Binary target after your negative downsampling.
        month_flat : 1D array
            Month for each sample in y.
        month_keep_frac : dict
            Example: {4: 0.4, 5: 0.6}
            means keep 40% of positives in April and 60% in May.
        rng : np.random.Generator
    
        Returns
        -------
        idx_keep : 1D sorted integer indices
            Indices to keep from the current dataset.
        """
        y = np.asarray(y)
        month_flat = np.asarray(month_flat)
    
        idx_all = np.arange(len(y))
        idx_neg = idx_all[y == 0]
        idx_pos = idx_all[y == 1]
    
        idx_pos_keep_parts = []
    
        for m in np.unique(month_flat[y == 1]):
            idx_pos_m = idx_pos[month_flat[idx_pos] == m]
    
            if m in month_keep_frac:
                frac = float(month_keep_frac[m])
                n_keep = int(np.round(len(idx_pos_m) * frac))
                n_keep = max(1, n_keep) if len(idx_pos_m) > 0 else 0
    
                if n_keep < len(idx_pos_m):
                    idx_keep_m = rng.choice(idx_pos_m, size=n_keep, replace=False)
                else:
                    idx_keep_m = idx_pos_m
            else:
                idx_keep_m = idx_pos_m
    
            idx_pos_keep_parts.append(idx_keep_m)
    
        if len(idx_pos_keep_parts) > 0:
            idx_pos_keep = np.sort(np.concatenate(idx_pos_keep_parts))
        else:
            idx_pos_keep = np.array([], dtype=int)
    
        idx_keep = np.sort(np.concatenate([idx_neg, idx_pos_keep]))
        return idx_keep
        
    def find_best_fbeta_threshold(y_true, y_prob, beta=1.0):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
        beta2 = beta ** 2
        fbeta_values = (
            (1 + beta2) * precision[:-1] * recall[:-1]
            / (beta2 * precision[:-1] + recall[:-1] + 1e-12)
        )
    
        best_idx = np.argmax(fbeta_values)
    
        best_thr = float(thresholds[best_idx])
        best_fbeta = float(fbeta_values[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
    
        return {
            "beta": float(beta),
            "best_thr": best_thr,
            "best_fbeta": best_fbeta,
            "best_precision": best_precision,
            "best_recall": best_recall,
        }

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
            "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
            "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),              
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

    def _sample_rows_for_shap(X, rng_seed=42, max_samples=50000):
        X = np.asarray(X, dtype=np.float32)
        finite_row = np.all(np.isfinite(X), axis=1)
        n_finite = int(finite_row.sum())
        n_total = X.shape[0]

        print(f"[SHAP] Finite rows: {n_finite:,}/{n_total:,} ({n_finite/max(1,n_total):.1%})")

        if n_finite < 10:
            return X[finite_row], None

        idx_finite = np.where(finite_row)[0]
        rng = np.random.default_rng(rng_seed)

        if n_finite > max_samples:
            sel = rng.choice(idx_finite, size=max_samples, replace=False)
            print(f"[SHAP] Sampling {max_samples:,}/{n_total:,} rows.")
            return X[sel], sel
        else:
            print(f"[SHAP] Using all {n_finite:,} finite rows.")
            return X[finite_row], idx_finite

    def save_shap_global_and_dependence(booster, X_explain, feature_names, plots_dir, stats_dir,
                                        fold, rng_seed=0, max_samples=50000, topk_dependence=4):
        X_use, _ = _sample_rows_for_shap(X_explain, rng_seed=rng_seed, max_samples=max_samples)
        if X_use.shape[0] < 10:
            print("[SHAP] Too few rows after filtering. Skipping SHAP.")
            return

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_use)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        out_csv = os.path.join(stats_dir, f"shap_mean_abs_fold{fold:02d}.csv")
        df_imp.to_csv(out_csv, index=False)
        print(f"[SHAP] Saved: {out_csv}")

        out_summary = os.path.join(plots_dir, f"shap_summary_fold{fold:02d}.png")
        plt.close("all")
        plt.figure(figsize=(8.0, 5.8))
        shap.summary_plot(shap_values, X_use, feature_names=feature_names, show=False, max_display=len(feature_names))
        plt.tight_layout()
        plt.savefig(out_summary, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved: {out_summary}")

        out_bar = os.path.join(plots_dir, f"shap_bar_fold{fold:02d}.png")
        plt.figure(figsize=(8.0, 5.2))
        shap.summary_plot(shap_values, X_use, feature_names=feature_names, plot_type="bar",
                          show=False, max_display=len(feature_names))
        plt.tight_layout()
        plt.savefig(out_bar, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved: {out_bar}")

        top_feats = df_imp["feature"].head(int(topk_dependence)).tolist()
        for feat in top_feats:
            out_dep = os.path.join(plots_dir, f"shap_dependence_{feat}_fold{fold:02d}.png")
            shap.dependence_plot(feat, shap_values, X_use, feature_names=feature_names, show=False)
            plt.gcf().set_size_inches(7.2, 5.2)
            plt.tight_layout()
            plt.savefig(out_dep, dpi=200, bbox_inches="tight")
            plt.close(plt.gcf())
            print(f"[SHAP] Saved: {out_dep}")

    # ============================================================
    # 8) Containers
    # ============================================================
    cv_results = []
    hail_stats_rows = []
    hail_reduction_rows = []
    day_stats_rows = []
    pr_curves_test_tuned = []
    roc_curves_test_tuned = []
    pr_curves_test_baseline = []
    roc_curves_test_baseline = []

    # ============================================================
    # 9) CV loop
    # ============================================================
    for ii, (tr_idx, val_idx, te_idx) in enumerate(zip(fold_train, fold_val, fold_test)):
        print("\n" + "=" * 70)
        print(
            f"FOLD {ii:02d} | "
            f"train_days={len(tr_idx)} | val_days={len(val_idx)} | test_days={len(te_idx)} | "
            f"thr={threshold_cm:.1f}cm | ds_factor_us={ds_factor_us}"
        )
        print("=" * 70)

        train_full_idx = np.sort(np.concatenate([tr_idx, val_idx]))
        all_idx = np.sort(np.concatenate([tr_idx, val_idx, te_idx]))

        dates_tr = pd.to_datetime(rgd_time_dd[tr_idx])
        dates_train = pd.to_datetime(rgd_time_dd[train_full_idx])
        dates_val = pd.to_datetime(rgd_time_dd[val_idx])
        dates_te = pd.to_datetime(rgd_time_dd[te_idx])
        dates_all = pd.to_datetime(rgd_time_dd[all_idx])

        # ------------------------------------------------------------
        # Fold-specific monthly climatology from training years only
        # ------------------------------------------------------------
        all_months_pos_us = []

        for t in tr_idx:
            m = rgd_time_dd[t].month

            hb_us = rgr_noaa_obs[0, t, lsm_us >= 0.5]
            sz_us = rgr_noaa_obs[1, t, lsm_us >= 0.5]
            n_pos_us = int(np.sum((hb_us == 1) & (sz_us >= threshold_cm)))
            all_months_pos_us.extend([m] * n_pos_us)

        clim_freq_us = monthly_clim_freq(np.array(all_months_pos_us, dtype=int), pseudocount=0.0)

        # ------------------------------------------------------------
        # Observation cubes
        # ------------------------------------------------------------
        hail_bin_us = rgr_noaa_obs[0, :, :, :]
        hail_size_us = rgr_noaa_obs[1, :, :, :]

        y_hb_train_us = np.array(hail_bin_us[train_full_idx, :, :]).astype(float)
        y_sz_train_us = np.array(hail_size_us[train_full_idx, :, :]).astype(float)
        y_hb_tr_us = np.array(hail_bin_us[tr_idx, :, :]).astype(float)
        y_sz_tr_us = np.array(hail_size_us[tr_idx, :, :]).astype(float)
        y_hb_val_us = np.array(hail_bin_us[val_idx, :, :]).astype(float)
        y_sz_val_us = np.array(hail_size_us[val_idx, :, :]).astype(float)
        y_hb_te_us = np.array(hail_bin_us[te_idx, :, :]).astype(float)
        y_sz_te_us = np.array(hail_size_us[te_idx, :, :]).astype(float)

        y_hb_all_us = np.array(hail_bin_us[all_idx, :, :]).astype(float)
        y_sz_all_us = np.array(hail_size_us[all_idx, :, :]).astype(float)

        for arr in (y_hb_train_us, y_sz_train_us, y_hb_tr_us, y_sz_tr_us, y_hb_val_us, y_sz_val_us, y_hb_te_us, y_sz_te_us):
            arr[:, lsm_us < 0.5] = np.nan
            
        y_hb_all_us[:, lsm_us < 0.5] = np.nan
        y_sz_all_us[:, lsm_us < 0.5] = np.nan

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
        
        land_mask_all_us, year_flat_all_us, month_flat_all_us, hb_flat_all_us, sz_flat_all_us, valid_binary_all_us, y_all_us = \
            prepare_binary_obs(y_hb_all_us, y_sz_all_us, dates_all, threshold_cm)

        # Stats
        add_hail_stats_rows(hail_stats_rows, ii, "US_tr_noDS", month_flat_tr_us, hb_flat_tr_us, sz_flat_tr_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_train_noDS", month_flat_train_us, hb_flat_train_us, sz_flat_train_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_val_noDS", month_flat_val_us, hb_flat_val_us, sz_flat_val_us, threshold_cm)
        add_hail_stats_rows(hail_stats_rows, ii, "US_test_noDS", month_flat_te_us, hb_flat_te_us, sz_flat_te_us, threshold_cm)

        # Day-level stats
        event_day_tr_us = np.any((y_hb_tr_us == 1) & (y_sz_tr_us >= threshold_cm), axis=(1, 2))
        event_day_val_us = np.any((y_hb_val_us == 1) & (y_sz_val_us >= threshold_cm), axis=(1, 2))

        add_daylevel_stats_rows(day_stats_rows, ii, "US_tr_days", dates_tr, event_day_tr_us)
        add_daylevel_stats_rows(day_stats_rows, ii, "US_val_days", dates_val, event_day_val_us)

        # ------------------------------------------------------------
        # Negative downsampling
        # ------------------------------------------------------------
        idx_keep_tr_us, idx_no_us_raw, sel_no_us = downsample_negatives(y_tr_us, ds_factor_us, np.random.default_rng(123 + ii))
        idx_keep_train_us, idx_no_train_us_raw, sel_no_train_us = downsample_negatives(y_train_us, ds_factor_us, np.random.default_rng(223 + ii))
        idx_keep_all_us, _, _ = downsample_negatives(y_all_us, ds_factor_us, np.random.default_rng(9000 + ii))

        y_tr_bal_us = y_tr_us[idx_keep_tr_us]
        y_train_bal_us = y_train_us[idx_keep_train_us]
        y_all_bal_us = y_all_us[idx_keep_all_us]

        neg_mask_all_us = np.zeros_like(hb_flat_tr_us, dtype=bool)
        neg_mask_all_us[idx_no_us_raw] = True
        neg_mask_kept_us = np.zeros_like(hb_flat_tr_us, dtype=bool)
        neg_mask_kept_us[sel_no_us] = True

        neg_mask_all_train_us = np.zeros_like(hb_flat_train_us, dtype=bool)
        neg_mask_all_train_us[idx_no_train_us_raw] = True
        neg_mask_kept_train_us = np.zeros_like(hb_flat_train_us, dtype=bool)
        neg_mask_kept_train_us[sel_no_train_us] = True

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

        # ------------------------------------------------------------
        # Predictor flattening
        # ------------------------------------------------------------
        X_grid_tr_us = np.copy(rgr_era_varall_us[tr_idx, :, :, :]).astype(np.float32)
        X_grid_train_us = np.copy(rgr_era_varall_us[train_full_idx, :, :, :]).astype(np.float32)
        X_grid_val_us = np.copy(rgr_era_varall_us[val_idx, :, :, :]).astype(np.float32)
        X_grid_te_us = np.copy(rgr_era_varall_us[te_idx, :, :, :]).astype(np.float32)
        X_grid_all_us = np.copy(rgr_era_varall_us[all_idx, :, :, :]).astype(np.float32)

        X_grid_tr_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_train_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_val_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_te_us[:, :, lsm_us < 0.5] = np.nan
        X_grid_all_us[:, :, lsm_us < 0.5] = np.nan

        X_tr_us_bin = flatten_predictors(X_grid_tr_us, land_mask_tr_us, valid_binary_tr_us)
        X_train_us_bin = flatten_predictors(X_grid_train_us, land_mask_train_us, valid_binary_train_us)
        X_val_us_bin = flatten_predictors(X_grid_val_us, land_mask_val_us, valid_binary_val_us)
        X_te_us_bin = flatten_predictors(X_grid_te_us, land_mask_te_us, valid_binary_te_us)
        X_all_us_bin = flatten_predictors(X_grid_all_us, land_mask_all_us, valid_binary_all_us)

        X_tr_bal_us = X_tr_us_bin[idx_keep_tr_us, :]
        X_train_bal_us = X_train_us_bin[idx_keep_train_us, :]
        X_all_bal_us = X_all_us_bin[idx_keep_all_us, :]

        month_tr_bal_us = month_flat_tr_us[idx_keep_tr_us]
        year_tr_bal_us = year_flat_tr_us[idx_keep_tr_us]
        month_train_bal_us = month_flat_train_us[idx_keep_train_us]
        year_train_bal_us = year_flat_train_us[idx_keep_train_us]
        month_all_bal_us = month_flat_all_us[idx_keep_all_us]

        # ------------------------------------------------------------
        # Additional Downsampling of positives
        # ------------------------------------------------------------
        rng_pos_us_tr = np.random.default_rng(1000 + ii)
        rng_pos_us_train = np.random.default_rng(2000 + ii)
        
        pos_keep_frac_us = {4: 0.35} 
        
        idx_keep_pos_tr_us = downsample_positive_months(
            y_tr_bal_us, month_tr_bal_us, pos_keep_frac_us, rng_pos_us_tr
        )
        idx_keep_pos_train_us = downsample_positive_months(
            y_train_bal_us, month_train_bal_us, pos_keep_frac_us, rng_pos_us_train
        )
        idx_keep_pos_all_us = downsample_positive_months(
            y_all_bal_us, month_all_bal_us, pos_keep_frac_us, np.random.default_rng(9200 + ii)
        )

        # comment out when positive downsampling should NOT be applied
        
        X_tr_bal_us = X_tr_bal_us[idx_keep_pos_tr_us, :]
        y_tr_bal_us = y_tr_bal_us[idx_keep_pos_tr_us]
        month_tr_bal_us = month_tr_bal_us[idx_keep_pos_tr_us]

        X_train_bal_us = X_train_bal_us[idx_keep_pos_train_us, :]
        y_train_bal_us = y_train_bal_us[idx_keep_pos_train_us]
        month_train_bal_us = month_train_bal_us[idx_keep_pos_train_us]

        X_all_bal_us = X_all_bal_us[idx_keep_pos_all_us, :]
        y_all_bal_us = y_all_bal_us[idx_keep_pos_all_us]
        month_all_bal_us = month_all_bal_us[idx_keep_pos_all_us]
        
        # ------------------------------------------------------------
        # Dataset assignments
        # ------------------------------------------------------------
        X_tr_bal = X_tr_bal_us
        y_tr_bal = y_tr_bal_us

        X_train_bal = X_train_bal_us
        y_train_bal = y_train_bal_us

        X_tr = X_tr_us_bin
        y_tr = y_tr_us

        X_val = X_val_us_bin
        y_val = y_val_us

        X_test = X_te_us_bin
        y_test = y_test_us

        X_all_bal = X_all_bal_us
        y_all_bal = y_all_bal_us

        # ------------------------------------------------------------        
        # targeted sample weights
        sw_tr_us = compute_targeted_weights_us(y_tr_bal_us, month_tr_bal_us)
        sw_train_us = compute_targeted_weights_us(y_train_bal_us, month_train_bal_us)
        sw_all_us = compute_targeted_weights_us(y_all_bal_us, month_all_bal_us)
        
        sw_tr_us *= target_us_share
        sw_train_us *= target_us_share
        sw_all_us *= target_us_share

        sw_tr = sw_tr_us / np.mean(sw_tr_us)
        sw_train = sw_train_us / np.mean(sw_train_us)
        sw_all = sw_all_us / np.mean(sw_all_us)

        print(f"Fold {ii:02d} | sample weights: mean={np.mean(sw_tr):.4f}, min={np.min(sw_tr):.4f}, max={np.max(sw_tr):.4f}")

        # ------------------------------------------------------------
        # More stats
        # ------------------------------------------------------------
        add_hail_stats_rows(
            hail_stats_rows, ii, "US_tr_DS",
            month_flat_tr_us[idx_keep_tr_us], hb_flat_tr_us[idx_keep_tr_us], sz_flat_tr_us[idx_keep_tr_us],
            threshold_cm,
            ds_ratio=(np.sum(y_tr_bal_us == 0) / max(1, np.sum(y_tr_bal_us == 1)))
        )

        add_hail_stats_rows(
            hail_stats_rows, ii, "US_train_DS",
            month_flat_train_us[idx_keep_train_us], hb_flat_train_us[idx_keep_train_us], sz_flat_train_us[idx_keep_train_us],
            threshold_cm,
            ds_ratio=(np.sum(y_train_bal_us == 0) / max(1, np.sum(y_train_bal_us == 1)))
        )

        # ------------------------------------------------------------
        # DMatrices with sample weights
        # ------------------------------------------------------------
        dtr_balanced = xgb.DMatrix(
            X_tr_bal, label=y_tr_bal, weight=sw_tr,
            missing=np.nan, feature_names=feature_names
        )
        dtrain_balanced = xgb.DMatrix(
            X_train_bal, label=y_train_bal, weight=sw_train,
            missing=np.nan, feature_names=feature_names
        )

        dall_balanced = xgb.DMatrix(
            X_all_bal, label=y_all_bal, weight=sw_all,
            missing=np.nan, feature_names=feature_names,
        )
        
        dtr_imbalanced = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_names)
        dval_imbalanced = xgb.DMatrix(X_val, label=y_val, missing=np.nan, feature_names=feature_names)
        dte = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=feature_names)

        # ------------------------------------------------------------
        # Scale_pos_weight (not applied)
        # ------------------------------------------------------------
        # spw_tr = (y_tr_bal == 0).sum() / max(1, (y_tr_bal == 1).sum())
        # spw_train = (y_train_bal == 0).sum() / max(1, (y_train_bal == 1).sum())

        # spw_tr_damped = 0.01 * spw_tr
        # spw_train_damped = 0.01 * spw_train

        # ------------------------------------------------------------
        # Training Baseline model
        # ------------------------------------------------------------
        base_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "device": "cuda",
            "max_bin": 64,
            "seed": 42 + ii,
        }

        default_params = {
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "alpha": 1.0,
            "lambda": 1.0,
            "gamma": 0.0,
        }

        baseline_params = {**base_params, **default_params}

        evals_result = {}
        model_baseline = xgb.train(
            params=baseline_params,
            dtrain=dtr_balanced,
            num_boost_round=200,
            evals=[(dtr_balanced, "train"), (dval_imbalanced, "valid")],
            early_stopping_rounds=40,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iter_baseline = model_baseline.best_iteration
        xgb_aucpr_valid_untuned = float(model_baseline.best_score)
        
        # ------------------------------------------------------------
        # final BASELINE model, retrained
        # ------------------------------------------------------------
        model_baseline_final = xgb.train(
            params=baseline_params,
            dtrain=dtrain_balanced,
            num_boost_round=int(best_iter_baseline) + 1,
            verbose_eval=False,
        )

        print("\n=== Summary ===")
        print(f"Best iteration, baseline model: {best_iter_baseline}")
        print(f"Best AUCPR, baseline model: {xgb_aucpr_valid_untuned:.6f}")

        # ------------------------------------------------------------
        # Training Tuned model
        # ------------------------------------------------------------

        fixed_params = {
            'learning_rate': 0.047507773394883004,            
            'max_depth': 5,            
            'min_child_weight':8.818188005763513,            
            'subsample': 0.6038094208107587,
            'colsample_bytree': 0.5029985914779086 ,            
            'alpha': 1.1061362023365987e-06,            
            'gamma': 2.184307346178722,            
            'lambda': 0.014885192049452677,
        }
        
        tuned_params = {**base_params, **fixed_params}

        evals_result = {}
        model_tuned = xgb.train(
            params=tuned_params,
            dtrain=dtr_balanced,
            num_boost_round=200,
            evals=[(dtr_balanced, "train"), (dval_imbalanced, "valid")],
            early_stopping_rounds=40,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iter_tuned = model_tuned.best_iteration
        xgb_aucpr_valid_tuned = float(model_tuned.best_score)

        # ------------------------------------------------------------
        # final TUNED model, retrained
        # ------------------------------------------------------------
        base_params_final = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "device": "cuda",
            "max_bin": 64,
            "seed": 42 + ii,
        }

        tuned_params_final = {**base_params_final, **fixed_params}

        model_tuned_final = xgb.train(
            params=tuned_params_final,
            dtrain=dtrain_balanced,
            num_boost_round=int(best_iter_tuned) + 1,
            verbose_eval=False,
        )

        print("\n=== Summary ===")
        print(f"Best iteration: {best_iter_tuned}")
        print(f"Best AUCPR: {xgb_aucpr_valid_tuned:.6f}")
        print(f"Fixed params: {fixed_params}")

        # ------------------------------------------------------------
        # final TUNED model, retrained INCLUSIVE TEST DATA
        # ------------------------------------------------------------
        base_params_final = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "device": "cuda",
            "max_bin": 64,
            "seed": 42 + ii,
        }

        tuned_params_final = {**base_params_final, **fixed_params}

        model_dall = xgb.train(
            params=tuned_params_final,
            dtrain=dall_balanced,
            num_boost_round=int(best_iter_tuned) + 1,
            verbose_eval=False,
        )

        print("\n=== Summary ===")
        print(f"Best iteration: {best_iter_tuned}")
        print(f"Best AUCPR: {xgb_aucpr_valid_tuned:.6f}")
        print(f"Fixed params: {fixed_params}")

        # ------------------------------------------------------------
        # Predictions on test data with BASELINE MODEL
        # ------------------------------------------------------------
        y_test_prob_baseline = model_baseline_final.predict(dte)

        prec_test_baseline, rec_test_baseline, _ = precision_recall_curve(y_test, y_test_prob_baseline)
        aucpr_test_baseline = auc(rec_test_baseline, prec_test_baseline)
        sklearn_ap_test_baseline = average_precision_score(y_test, y_test_prob_baseline)

        fpr_test_baseline, tpr_test_baseline, _ = roc_curve(y_test, y_test_prob_baseline)
        aucroc_test_baseline = roc_auc_score(y_test, y_test_prob_baseline)

        pr_curves_test_baseline.append({
            "fold": ii,
            "recall": rec_test_baseline,
            "precision": prec_test_baseline,
            "aucpr": aucpr_test_baseline,
            "baseline": np.mean(y_test),
        })

        roc_curves_test_baseline.append({
            "fold": ii,
            "fpr": fpr_test_baseline,
            "tpr": tpr_test_baseline,
            "roc_auc": aucroc_test_baseline,
        })

        # ------------------------------------------------------------
        # Predictions on test data with FINAL MODEL
        # ------------------------------------------------------------
        y_test_prob = model_tuned_final.predict(dte)

        precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
        aucpr_test = auc(recall_test, precision_test)
        sklearn_ap_test = average_precision_score(y_test, y_test_prob)

        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
        aucroc_test = roc_auc_score(y_test, y_test_prob)

        # metrics at fixed threshold = 0.5
        thr_metrics_test = compute_threshold_ranking_metrics(
            y_test, y_test_prob, thr=0.5
        )

        # thresholds that maximize F0.5, F1, and F2 on test data
        best_f05_test = find_best_fbeta_threshold(y_test, y_test_prob, beta=0.5)
        best_f1_test = find_best_fbeta_threshold(y_test, y_test_prob, beta=1.0)
        best_f2_test = find_best_fbeta_threshold(y_test, y_test_prob, beta=2.0)


        # confusion-matrix-based metrics at those thresholds
        thr_metrics_test_best_f05 = compute_threshold_ranking_metrics(
            y_test, y_test_prob, thr=best_f05_test["best_thr"]
        )
        
        thr_metrics_test_best_f1 = compute_threshold_ranking_metrics(
            y_test, y_test_prob, thr=best_f1_test["best_thr"]
        )
        
        thr_metrics_test_best_f2 = compute_threshold_ranking_metrics(
            y_test, y_test_prob, thr=best_f2_test["best_thr"]
        )
        
        pr_curves_test_tuned.append({
            "fold": ii,
            "recall": recall_test,
            "precision": precision_test,
            "aucpr": aucpr_test,
            "baseline": np.mean(y_test),
        })

        roc_curves_test_tuned.append({
            "fold": ii,
            "fpr": fpr_test,
            "tpr": tpr_test,
            "roc_auc": aucroc_test,
        })
        # ------------------------------------------------------------
        # Foldwise PR-Curves for Overfitting Check (preds. on imbalanced Train and Test
        # ------------------------------------------------------------
        # prediction on train data
        global_prevalence_tr_imbalanced = float(np.mean(y_tr))
        global_prevalence_test = float(np.mean(y_test))

        y_train_pred = model_tuned_final.predict(dtr_imbalanced)
        precision_train, recall_train, _ = precision_recall_curve(y_tr, y_train_pred)
        aucpr_train = auc(recall_train, precision_train)
        sklearn_ap_train = average_precision_score(y_tr, y_train_pred)        

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(
            recall_train, precision_train,
            linewidth=0.8,
            color="#588157",
            label=f"Train PR curve (AUC={aucpr_train:.4f})"
        )
        ax.plot(
            recall_test, precision_test,
            linewidth=0.8,
            color="#dda15e",
            label=f"Test PR curve (AUC={aucpr_test:.4f})"
        )

        ax.axhline(
            global_prevalence_test,
            linestyle=":",
            linewidth=2,
            color="#dda15e",
            label=f"Test Baseline (prevalence={global_prevalence_test:.4f})"
        )
        ax.axhline(
            global_prevalence_tr_imbalanced,
            linestyle=":",
            linewidth=2,
            color="#588157",
            label=f"Train Baseline (prevalence={global_prevalence_tr_imbalanced:.4f})"
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
        # CV results
        # ------------------------------------------------------------
        cv_results.append({
            "fold": ii,
            "train_years_full": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[train_full_idx]).year))),
            "train_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[tr_idx]).year))),
            "val_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[val_idx]).year))),
            "test_years": ",".join(map(str, np.unique(pd.DatetimeIndex(rgd_time_dd[te_idx]).year))),

            "global_prevalence_tr": np.mean(y_tr_bal),
            "global_prevalence_tr_imbalanced": np.mean(y_tr),
            "global_prevalence_valid": np.mean(y_val),
            "global_prevalence_train": np.mean(y_train_bal),
            "global_prevalence_test": np.mean(y_test),

            # main metrics (tuned model)          
            "tp": thr_metrics_test["tp"],
            "tn": thr_metrics_test["tn"],
            "fp": thr_metrics_test["fp"],
            "fn": thr_metrics_test["fn"],
            "tp_f1max": thr_metrics_test_best_f1["tp"],
            "tn_f1max": thr_metrics_test_best_f1["tn"],
            "fp_f1max": thr_metrics_test_best_f1["fp"],
            "fn_f1max": thr_metrics_test_best_f1["fn"],
            
            "n_pos_test": int(np.sum(y_test == 1)),
            "n_neg_test": int(np.sum(y_test == 0)),
            
            "acc_thr": thr_metrics_test["accuracy"],
            "prec_thr": thr_metrics_test["precision"],
            "rec_thr": thr_metrics_test["recall"],
            
            "acc_f1max": thr_metrics_test_best_f1["accuracy"],            
            "prec_test_f1max": best_f1_test["best_precision"],
            "rec_test_f1max": best_f1_test["best_recall"],

            "thr_test": thr_metrics_test["thr"],
            "thr_test_f05max": best_f05_test["best_thr"],
            "thr_test_f1max": best_f1_test["best_thr"],
            "thr_test_f2max": best_f2_test["best_thr"],
            "f05_max": best_f05_test["best_fbeta"],
            "f1_max": best_f1_test["best_fbeta"],
            "f2_max": best_f2_test["best_fbeta"],
            "f05_thr": thr_metrics_test["f0.5"],
            "f1_thr": thr_metrics_test["f1"],
            "f2_thr": thr_metrics_test["f2"],
            
            "mcc_thr": thr_metrics_test["mcc"],
            "kappa": thr_metrics_test["kappa"],
            "mcc_f1max": thr_metrics_test_best_f1["mcc"],
            "kappa_f1max": thr_metrics_test_best_f1["kappa"],
            
            "roc_auc_tuned": float(aucroc_test),
            "pr_auc_tuned": float(sklearn_ap_test),
 
            # main metrics (baseline model)
            "roc_auc_baseline": float(aucroc_test_baseline),
            "pr_auc_baseline": float(sklearn_ap_test_baseline),
        })

        # ------------------------------------------------------------
        # SHAP
        # ------------------------------------------------------------
        try:
            save_shap_global_and_dependence(
                booster=model_tuned_final,
                X_explain=X_test,
                feature_names=feature_names,
                plots_dir=str(plots_dir),
                stats_dir=str(stats_dir),
                fold=ii,
                rng_seed=777 + ii,
                max_samples=50000,
                topk_dependence=8,
            )
        except Exception as exc:
            print(f"[SHAP] Failed on fold {ii:02d}: {exc}")
            traceback.print_exc()

        # ------------------------------------------------------------
        # Save fold outputs
        # ------------------------------------------------------------
        model_path = os.path.join(models_dir, f"xgbUS_sigHail_fold{ii}.json")
        model_path_dall = os.path.join(models_dir, f"xgbUS_sigHail_FINAL_2000-2022_fold{ii}.json")
        
        model_tuned_final.save_model(model_path)
        model_dall.save_model(model_path_dall)
        
        np.save(os.path.join(preds_dir, f"fold{ii}_y_test_prob.npy"), y_test_prob)

    # ============================================================
    # 10) Combined plots
    # ============================================================
    plot_combined_pr_curves(
        curves=pr_curves_test_tuned,
        outpath=os.path.join(plots_dir, "pr_curves_all_folds_test_tuned.png"),
        title="Precision-Recall Curves on Test Data (all CV folds)",
        ylim=(0, 0.1),
    )

    plot_combined_roc_curves(
        curves=roc_curves_test_tuned,
        outpath=os.path.join(plots_dir, "roc_curves_all_folds_test_tuned.png"),
        title="ROC Curves on Test Data (all CV folds)",
    )

    plot_combined_pr_curves(
        curves=pr_curves_test_baseline,
        outpath=os.path.join(plots_dir, "pr_curves_all_folds_test_baseline.png"),
        title="Precision-Recall Curves on Test Data (all CV folds)",
        ylim=(0, 0.1),
    )

    plot_combined_roc_curves(
        curves=roc_curves_test_baseline,
        outpath=os.path.join(plots_dir, "roc_curves_all_folds_test_baseline.png"),
        title="ROC Curves on Test Data (all CV folds)",
    )

    # ============================================================
    # 11) Save tables
    # ============================================================
    df_cv = pd.DataFrame(cv_results)
    print(
        f"[CV] AUCPR TEST max={df_cv['pr_auc_tuned'].max():.6f} | "
        f"median={df_cv['pr_auc_tuned'].median():.6f} | "
        f"mean={df_cv['pr_auc_tuned'].mean():.6f} | "
        f"std={df_cv['pr_auc_tuned'].std():.6f}"
    )

    df_cv.to_csv(os.path.join(stats_dir, "cv_results_all_folds.csv"), index=False)
    pd.DataFrame(hail_stats_rows).to_csv(os.path.join(stats_dir, "hail_gridcell_days.csv"), index=False)
    pd.DataFrame(hail_reduction_rows).to_csv(os.path.join(stats_dir, "hail_gridcell_days_reduction.csv"), index=False)
    pd.DataFrame(day_stats_rows).to_csv(os.path.join(stats_dir, "hail_days_train_val.csv"), index=False)

    print("[SAVE] Finished.")


if __name__ == "__main__":
    main()