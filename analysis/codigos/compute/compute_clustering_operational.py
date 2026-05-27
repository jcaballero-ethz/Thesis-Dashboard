"""
Operational Feature Extraction and Clustering
==============================================
Extracts operational anomaly features for each stress event and clusters them using
K-Medoids, for both the future and historical datasets.

Features (39 per event):
  For each of 13 geographic zones: net transfer anomaly, storage level anomaly,
  storage discharge anomaly
  Anomaly = event_value - cross-scenario baseline mean  (absolute, GWh units)

Method:
  1. Load filtered stress events (cost_share ≥ 0.2%, duration ≥ 1 day)
  2. For each event, compute zone-level operational anomalies relative to the
     mean across all 60 scenarios
  3. Scale features (StandardScaler) → reduce with PCA (90% variance threshold)
  4. Select optimal k via silhouette score, then fit K-Medoids
  5. Save clustered feature CSV and diagnostic plots

Inputs:
  - ZEN-garden output folders (future + historical)
  - events_global_alpha{ALPHA:02d}.csv / events_global_historical_alpha{ALPHA:02d}.csv

Outputs:
  - CSVs/future/features/features_alpha25_operational_future_clustered.csv
  - CSVs/historical/features/features_alpha25_operational_historical_clustered.csv
  - Silhouette and PCA scatter plots (saved to home directory)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: CSV data + PNG figures (silhouette plot, PCA scatter)
ALPHA      = 25     # cost-coverage threshold used when selecting events
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to keep an event
THRESH_DUR = 1.0   # minimum duration (days) to keep an event
ALL_SC     = list(range(1, 61))
STOR_TECH  = ['battery', 'hydrogen_storage', 'pumped_hydro', 'reservoir_hydro']
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
# Results(path)                       — loads all scenario outputs
# r.get_full_ts('flow_transport')     — power line flows per (technology, edge)
#                                       index: ('power_line', 'A-B'), columns: hours 0..8759
#                                       values: GWh/h  (positive = A→B direction)
# r.get_full_ts('flow_transport_loss')— transmission losses on each line  [GWh/h]
# r.get_full_ts('storage_level')      — stored energy per (technology, node)  [GWh]
# r.get_full_ts('flow_storage_discharge') — discharge power per (technology, node)  [GWh/h]
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
        'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_operational_future.csv'),
        'out_csv':    os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv'),
        'sil_png':    os.path.expanduser('~/silhouette_operational_future.png'),
        'pca_png':    os.path.expanduser('~/pca_operational_future.png'),
        'zones':      ZONES_FUTURE,
    },
    'historical': {
        'path':       os.path.expanduser('~/Desktop/ZEN-garden model/outputs/20260202_GF_historical_with_cooling'),
        'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv'),
        'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_operational_historical.csv'),
        'out_csv':    os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv'),
        'sil_png':    os.path.expanduser('~/silhouette_operational_historical.png'),
        'pca_png':    os.path.expanduser('~/pca_operational_historical.png'),
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

        all_nodes     = [n for nodes in ZONES.values() for n in nodes]
        all_nodes_set = set(all_nodes)

        # Accumulators: baseline sums (across all scenarios) and event-specific values
        bl_nettrans   = np.zeros((n_ev, N_ZONES))
        bl_stor_diff  = np.zeros((n_ev, N_ZONES))
        bl_stor_disch = np.zeros((n_ev, N_ZONES))
        bl_count      = np.zeros(n_ev)
        ev_nettrans   = np.zeros((n_ev, N_ZONES))
        ev_stor_diff  = np.zeros((n_ev, N_ZONES))
        ev_stor_disch = np.zeros((n_ev, N_ZONES))

        for sc_all in ALL_SC:
            print(f'    sc{sc_all}...', flush=True)
            sc_name = f'scenario_{sc_all}'

            # Load once per scenario, filter with masks — avoids redundant ZEN-garden reads per event.
            ft_all = r.get_full_ts('flow_transport',         scenario_name=sc_name)
            fl_all = r.get_full_ts('flow_transport_loss',    scenario_name=sc_name)
            sl_all = r.get_full_ts('storage_level',          scenario_name=sc_name)
            fd_all = r.get_full_ts('flow_storage_discharge', scenario_name=sc_name)

            # Pre-aggregate net transport per node: imports positive, exports negative
            net_trans_node = {n: np.zeros(ft_all.shape[1]) for n in all_nodes}
            for idx in ft_all.index:
                if len(idx) == 2 and idx[0] == 'power_line' and '-' in idx[1]:
                    a, b = idx[1].split('-', 1)
                    flow = ft_all.loc[idx].values.astype(float)
                    try:
                        loss = fl_all.loc[idx].values.astype(float)
                    except KeyError:
                        loss = np.zeros_like(flow)
                    if a in all_nodes_set:
                        net_trans_node[a] -= flow
                    if b in all_nodes_set:
                        net_trans_node[b] += flow - loss

            # Pre-aggregate storage level and discharge per (tech, node)
            sl_node = {}
            fd_node = {}
            for tech in STOR_TECH:
                for n in all_nodes:
                    try:
                        sl_node[(tech, n)] = sl_all.loc[(tech, n)].values.astype(float)
                    except KeyError:
                        sl_node[(tech, n)] = None
                    try:
                        fd_node[(tech, n)] = fd_all.loc[(tech, n)].values.astype(float)
                    except KeyError:
                        fd_node[(tech, n)] = None

            for ev_idx, row in df_ev.iterrows():
                t_s   = int(row['t_start'])
                t_e   = int(row['t_end'])
                hrs   = list(range(t_s, t_e + 1))
                ev_sc = int(row['sc'])
                bl_count[ev_idx] += 1

                for zi, (zone, nodes) in enumerate(ZONES.items()):
                    nt_z = sum(float(np.mean(net_trans_node[n][hrs])) for n in nodes)
                    bl_nettrans[ev_idx, zi] += nt_z
                    if sc_all == ev_sc:
                        ev_nettrans[ev_idx, zi] = nt_z

                    # Storage level change (end - start) and mean discharge, summed over techs
                    sd_z = 0.0
                    fd_z = 0.0
                    for tech in STOR_TECH:
                        for n in nodes:
                            sm = sl_node.get((tech, n))
                            if sm is not None:
                                sd_z += sm[t_e] - sm[t_s]
                            fm = fd_node.get((tech, n))
                            if fm is not None:
                                fd_z += float(np.mean(fm[hrs]))
                    bl_stor_diff[ev_idx, zi]  += sd_z
                    bl_stor_disch[ev_idx, zi] += fd_z
                    if sc_all == ev_sc:
                        ev_stor_diff[ev_idx, zi]  = sd_z
                        ev_stor_disch[ev_idx, zi] = fd_z

        # Compute absolute anomalies: event value − cross-scenario baseline mean
        bl = bl_count[:, np.newaxis]
        nettrans_anom       = ev_nettrans   - bl_nettrans  / bl
        stor_level_anom     = ev_stor_diff  - bl_stor_diff / bl
        stor_discharge_anom = ev_stor_disch - bl_stor_disch / bl

        feat_cols   = []
        feat_arrays = []
        for zi, zone in enumerate(ZONE_NAMES):
            feat_cols   += [f'nettrans anom {zone}', f'stor level anom {zone}', f'stor discharge anom {zone}']
            feat_arrays += [nettrans_anom[:, zi], stor_level_anom[:, zi], stor_discharge_anom[:, zi]]

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
    ax.set_title(f'Operational clustering — silhouette ({ds_name})')
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(cfg['sil_png'], dpi=150); plt.close()

    # ── Final clustering ──────────────────────────────────────────────────────
    km     = KMedoids(n_clusters=K_OPT, random_state=42, max_iter=300)
    labels = km.fit_predict(X_cl)
    df_feat['cluster_operational'] = labels

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
    ax.set_title(f'Operational clusters k={K_OPT} — {ds_name}')
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(cfg['pca_png'], dpi=150); plt.close()

    # ── Save clustered features ───────────────────────────────────────────────
    df_feat.to_csv(cfg['out_csv'], index=False)
    print(f'  Saved: {cfg["out_csv"]}')

print('\nDone.')
