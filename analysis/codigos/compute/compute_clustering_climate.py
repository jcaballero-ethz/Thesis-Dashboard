"""
Climate Feature Extraction and Clustering
==========================================
Extracts climate anomaly features for each stress event and clusters them using
K-Medoids, for both the future and historical datasets.

Features (39 per event):
  For each of 13 geographic zones: wind anomaly, PV anomaly, heat demand anomaly
  Anomaly = (event_value - scenario_mean) / scenario_mean × 100  (percentage)

Method:
  1. Load filtered stress events (cost_share ≥ 0.2%, duration ≥ 1 day)
  2. For each event, compute zone-level climate anomalies relative to the
     scenario-wide mean (averaged across all 60 scenarios as baseline)
  3. Scale features (StandardScaler) → reduce with PCA (90% variance threshold)
  4. Select optimal k via silhouette score, then fit K-Medoids
  5. Save clustered feature CSV and diagnostic plots

Inputs:
  - ZEN-garden output folders (future + historical)
  - events_global_alpha{ALPHA:02d}.csv / events_global_historical_alpha{ALPHA:02d}.csv

Outputs:
  - CSVs/future/features/features_alpha25_climate_future_clustered.csv
  - CSVs/historical/features/features_alpha25_climate_historical_clustered.csv
  - Silhouette and PCA scatter plots (saved to home directory)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: CSV data + PNG figures (silhouette plot, PCA scatter)
ALPHA      = 25     # cost-coverage threshold used when selecting events
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to keep an event
THRESH_DUR = 1.0   # minimum duration (days) to keep an event
ALL_SC     = list(range(1, 61))
USE_PCA    = True
PCA_VAR    = 0.90  # fraction of variance to retain in PCA
K_RANGE    = range(2, 9)
K_OPT      = 4     # number of clusters (chosen by silhouette analysis)
# ──────────────────────────────────────────────────────────────────────────────

import logging
logging.getLogger("pint").setLevel(logging.ERROR)
import os, sys
sys.stderr = open(os.devnull, 'w')
from zen_garden import Results
sys.stderr.close()
sys.stderr = sys.__stderr__

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    raise ImportError("Run: pip install scikit-learn-extra")

# ── ZEN-garden API note ───────────────────────────────────────────────────────
# Results(path)               — loads all scenario outputs from the ZEN-garden folder
# r.get_full_ts('max_load')   — capacity factor time series per (technology, node)
#                               index: (technology, node), columns: hours 0..8759
#                               values: dimensionless [0–1]  (fraction of installed capacity)
# r.get_full_ts('demand')     — exogenous demand per (carrier, node)
#                               index: (carrier, node), columns: hours 0..8759
#                               values: GWh per hour
# ─────────────────────────────────────────────────────────────────────────────

# ── Zone definitions ──────────────────────────────────────────────────────────
# 13 geographic zones derived from Ward clustering on pairwise wind correlations
ZONES_FUTURE = {
    'Nordics_N':     ['FI', 'NO'],
    'Baltics':       ['EE', 'LV'],
    'Nordics_S':     ['DK', 'SE'],
    'British_Isles': ['IE', 'UK'],
    'NW_Continent':  ['DE', 'BE', 'NL'],
    'Med_East_N':    ['BG', 'RO'],
    'Med_East_S':    ['HR', 'SI'],
    'Greece':        ['EL'],
    'Italy':         ['IT'],
    'Iberia':        ['ES', 'PT'],
    'Central_W':     ['CH', 'FR'],
    'East_N':        ['LT', 'PL'],
    'Central_E':     ['HU', 'SK', 'AT', 'CZ'],
}

ZONES_HISTORICAL = {
    'Norway':        ['NO'],
    'Baltics_FI':    ['EE', 'LV', 'FI'],
    'Nordics_S':     ['DK', 'SE'],
    'British_Isles': ['IE', 'UK'],
    'NW_Continent':  ['DE', 'BE', 'NL'],
    'Med_East_N':    ['BG', 'RO'],
    'Med_East_S':    ['HR', 'SI'],
    'Greece':        ['EL'],
    'Italy':         ['IT'],
    'Iberia':        ['ES', 'PT'],
    'Central_W':     ['CH', 'FR'],
    'East_N':        ['LT', 'PL'],
    'Central_E':     ['HU', 'SK', 'AT', 'CZ'],
}

# ── Dataset configurations ────────────────────────────────────────────────────
DATASETS = {
    'future': {
        'path':       os.path.expanduser('~/Desktop/ZEN-garden model/outputs/20260202_GF_future_with_cooling'),
        'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha{ALPHA:02d}.csv'),
        'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_climate_future.csv'),
        'out_csv':    os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv'),
        'sil_png':    os.path.expanduser('~/silhouette_climate_future.png'),
        'pca_png':    os.path.expanduser('~/pca_climate_future.png'),
        'zones':      ZONES_FUTURE,
    },
    'historical': {
        'path':       os.path.expanduser('~/Desktop/ZEN-garden model/outputs/20260202_GF_historical_with_cooling'),
        'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv'),
        'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_climate_historical.csv'),
        'out_csv':    os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv'),
        'sil_png':    os.path.expanduser('~/silhouette_climate_historical.png'),
        'pca_png':    os.path.expanduser('~/pca_climate_historical.png'),
        'zones':      ZONES_HISTORICAL,
    },
}

# ── Main loop: process each dataset ──────────────────────────────────────────
for ds_name, cfg in DATASETS.items():
    print(f'\n{"="*70}')
    print(f'  DATASET: {ds_name.upper()}')
    print(f'{"="*70}')

    ZONES      = cfg['zones']
    ZONE_NAMES = list(ZONES.keys())
    N_ZONES    = len(ZONE_NAMES)

    # Load and filter stress events
    df_ev = pd.read_csv(cfg['events_csv'])
    df_ev = df_ev[(df_ev['cost_share_cross'] >= THRESH_CS) &
                  (df_ev['duration_days']    >= THRESH_DUR)].reset_index(drop=True)
    n_ev  = len(df_ev)
    print(f'  {n_ev} events after filtering')

    # ── Feature computation (or load from cache if already computed) ──────────
    if os.path.exists(cfg['feat_csv']):
        print(f'  Loading cached features from {cfg["feat_csv"]}...')
        df_feat   = pd.read_csv(cfg['feat_csv'])
        meta      = ['sc', 't_start', 't_end']
        feat_cols = [c for c in df_feat.columns if c not in meta]
        X         = df_feat[feat_cols].values
        print(f'  Loaded: {X.shape}')
    else:
        print('  Computing features — loading ZEN-garden Results...')
        r = Results(path=cfg['path'])
        # max_load: capacity factor for each (technology, node) — iloc[hrs] selects hour columns
        # demand: exogenous demand for each (carrier, node)

        # Accumulators: baseline sums (across all scenarios) and event-specific values
        bl_wind  = np.zeros((n_ev, N_ZONES))
        bl_pv    = np.zeros((n_ev, N_ZONES))
        bl_heat  = np.zeros((n_ev, N_ZONES))
        bl_count = np.zeros(n_ev)
        ev_wind  = np.zeros((n_ev, N_ZONES))
        ev_pv    = np.zeros((n_ev, N_ZONES))
        ev_heat  = np.zeros((n_ev, N_ZONES))

        for sc_all in ALL_SC:
            print(f'    sc{sc_all}...', flush=True)
            sc_name = f'scenario_{sc_all}'
            # Load once per scenario, filter with masks — avoids redundant ZEN-garden reads per event.
            ml_all  = r.get_full_ts('max_load', scenario_name=sc_name)
            dem_all = r.get_full_ts('demand',   scenario_name=sc_name)

            for ev_idx, row in df_ev.iterrows():
                t_s   = int(row['t_start'])
                t_e   = int(row['t_end'])
                hrs   = list(range(t_s, t_e + 1))
                ev_sc = int(row['sc'])
                bl_count[ev_idx] += 1

                for zi, (zone, nodes) in enumerate(ZONES.items()):
                    wind_z = sum(
                        float(np.mean(ml_all.loc[('wind_onshore', n)].iloc[hrs].values.astype(float)))
                        for n in nodes if ('wind_onshore', n) in ml_all.index
                    )
                    pv_z = sum(
                        float(np.mean(ml_all.loc[('photovoltaics', n)].iloc[hrs].values.astype(float)))
                        for n in nodes if ('photovoltaics', n) in ml_all.index
                    )
                    heat_z = sum(
                        float(np.mean(dem_all.loc[('heat', n)].iloc[hrs].values.astype(float)))
                        for n in nodes if ('heat', n) in dem_all.index
                    )
                    bl_wind[ev_idx, zi] += wind_z
                    bl_pv[ev_idx, zi]   += pv_z
                    bl_heat[ev_idx, zi] += heat_z
                    # Store the value from the event's own scenario
                    if sc_all == ev_sc:
                        ev_wind[ev_idx, zi] = wind_z
                        ev_pv[ev_idx, zi]   = pv_z
                        ev_heat[ev_idx, zi] = heat_z

        # Compute percentage anomaly: (event - baseline_mean) / baseline_mean × 100
        bl = bl_count[:, np.newaxis]
        wind_anom = np.where(bl_wind > 0, (ev_wind - bl_wind/bl) / (bl_wind/bl) * 100, 0.0)
        pv_anom   = np.where(bl_pv   > 0, (ev_pv   - bl_pv  /bl) / (bl_pv  /bl) * 100, 0.0)
        heat_anom = np.where(bl_heat > 0, (ev_heat - bl_heat/bl) / (bl_heat/bl) * 100, 0.0)

        feat_cols   = []
        feat_arrays = []
        for zi, zone in enumerate(ZONE_NAMES):
            feat_cols   += [f'wind anom {zone}', f'pv anom {zone}', f'heat anom {zone}']
            feat_arrays += [wind_anom[:, zi], pv_anom[:, zi], heat_anom[:, zi]]

        X = np.column_stack(feat_arrays)
        df_feat = pd.DataFrame(X, columns=feat_cols)
        df_feat['sc']      = df_ev['sc'].values
        df_feat['t_start'] = df_ev['t_start'].values
        df_feat['t_end']   = df_ev['t_end'].values
        df_feat.to_csv(cfg['feat_csv'], index=False)
        print(f'  Features saved: {cfg["feat_csv"]}  shape: {X.shape}')

    # ── Scale → PCA ──────────────────────────────────────────────────────────
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=PCA_VAR, random_state=42)
    X_cl = pca.fit_transform(X_sc)
    print(f'  PCA: {pca.n_components_} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)')
    cumvar = 0.0
    for i, v in enumerate(pca.explained_variance_ratio_):
        cumvar += v
        print(f'    PC{i+1}: {v*100:.1f}%  (cumulative: {cumvar*100:.1f}%)')

    # ── Silhouette analysis to select k ──────────────────────────────────────
    print('  Silhouette analysis...')
    sil_scores = []
    for k in K_RANGE:
        labels = KMedoids(n_clusters=k, random_state=42, max_iter=300).fit_predict(X_cl)
        score  = silhouette_score(X_cl, labels)
        sil_scores.append(score)
        print(f'    k={k}: {score:.4f}')

    print(f'  Using k = {K_OPT}')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_RANGE), sil_scores, marker='o', color='steelblue', linewidth=2)
    ax.axvline(K_OPT, color='firebrick', linestyle='--', linewidth=1.2, label=f'k*={K_OPT}')
    ax.set_xlabel('k'); ax.set_ylabel('Silhouette score')
    ax.set_title(f'Climate clustering — silhouette ({ds_name})')
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(cfg['sil_png'], dpi=150); plt.close()

    # ── Final clustering ──────────────────────────────────────────────────────
    km     = KMedoids(n_clusters=K_OPT, random_state=42, max_iter=300)
    labels = km.fit_predict(X_cl)
    df_feat['cluster_climate'] = labels

    print(f'\n  Cluster summary:')
    sil_vals = silhouette_samples(X_cl, labels)
    for c in range(K_OPT):
        mask = labels == c
        mi   = km.medoid_indices_[c]
        ev_r = df_ev.iloc[mi]
        print(f'    Cluster {c} (n={mask.sum()}, sil={sil_vals[mask].mean():.3f})')
        print(f'      Medoid: sc{int(ev_r["sc"])}, t={int(ev_r["t_start"])}-{int(ev_r["t_end"])} '
              f'({ev_r["duration_days"]:.1f}d, {ev_r["cost_share_cross"]:.3f}%)')

    # ── PCA scatter plot (2D projection for visualization) ────────────────────
    X_vis  = PCA(n_components=2).fit_transform(X_sc)
    colors = plt.cm.tab10(np.linspace(0, 1, K_OPT))
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(K_OPT):
        mask = labels == c
        ax.scatter(X_vis[mask, 0], X_vis[mask, 1], color=colors[c], s=60, alpha=0.8, label=f'C{c}')
        ax.scatter(X_vis[km.medoid_indices_[c], 0], X_vis[km.medoid_indices_[c], 1],
                   color=colors[c], s=220, marker='*', edgecolors='black', linewidths=0.8)
    ax.set_title(f'Climate clusters k={K_OPT} — {ds_name}')
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(cfg['pca_png'], dpi=150); plt.close()

    # ── Save clustered features ───────────────────────────────────────────────
    df_feat.to_csv(cfg['out_csv'], index=False)
    print(f'  Saved: {cfg["out_csv"]}')

print('\nDone.')
