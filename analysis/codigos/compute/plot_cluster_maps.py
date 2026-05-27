"""
Cluster Medoid Maps — Static PNGs
===================================
Generates 12 PNG maps showing the medoid event for each cluster, one PNG per
(feature type × dataset) combination (6 feature types × 2 datasets = 12 PNGs).
Each PNG contains a 2×2 grid of choropleth maps for the 4 clusters.

Feature types:
  Climate:     wind anom, PV anom, heat demand anom
  Operational: net transfer anom, storage level anom, storage discharge anom

Requires a Natural Earth shapefile at the path specified by SHAPEFILE.
Colorscales replicate the Plotly RdBu stops used in the interactive HTML maps.

Inputs:
  CSVs/future/features/features_alpha25_climate_future_clustered.csv
  CSVs/future/features/features_alpha25_operational_future_clustered.csv
  CSVs/historical/features/features_alpha25_climate_historical_clustered.csv
  CSVs/historical/features/features_alpha25_operational_historical_clustered.csv

Outputs:
  htmls/fotos/future/medoid_{feature}_future.png    (6 PNGs)
  htmls/fotos/historical/medoid_{feature}_historical.png  (6 PNGs)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: PNG figures only  (12 PNGs: 6 feature types × 2 datasets)
SHAPEFILE = '~/Desktop/shapefiles/ne_10m_admin_0_countries.shp'
OUT_DIR   = '~/Desktop/Bachelor Thesis/analysis/fotos'
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

SHAPEFILE = os.path.expanduser(SHAPEFILE)
OUT_DIR   = os.path.expanduser(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, 'future'),     exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'historical'), exist_ok=True)

CODE_MAP = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC', 'ES': 'ESP',
    'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL',
    'IT': 'ITA', 'LT': 'LTU', 'LV': 'LVA', 'NL': 'NLD', 'NO': 'NOR',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR',
}

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

CLUSTER_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

# ── Colormaps — exact Plotly RdBu color stops ─────────────────────────────────
_plotly_rdbu = [
    (0.0,  (5,   10,  172)),
    (0.35, (106, 137, 247)),
    (0.5,  (190, 190, 190)),
    (0.6,  (220, 170, 132)),
    (0.7,  (230, 145, 90)),
    (1.0,  (178, 10,  28)),
]

# Operational (blue=negative, red=positive)
CMAP_OPS = mcolors.LinearSegmentedColormap.from_list(
    'plotly_rdbu',
    [(pos, tuple(v/255 for v in rgb)) for pos, rgb in _plotly_rdbu],
)

# Wind/PV (red=negative, blue=positive — reversed)
_rdbu_rev = sorted(
    [(1 - pos, tuple(v/255 for v in rgb)) for pos, rgb in _plotly_rdbu],
    key=lambda x: x[0],
)
CMAP_CF = mcolors.LinearSegmentedColormap.from_list('plotly_rdbu_r', _rdbu_rev)

# Heat demand (flat blue for <0, white→orange→dark_red for 0-100)
CMAP_HEAT = mcolors.LinearSegmentedColormap.from_list('heat', [
    (0.0,    mcolors.to_rgb('#3182bd')),
    (0.0099, mcolors.to_rgb('#3182bd')),
    (0.01,   mcolors.to_rgb('#fff7ec')),
    (0.35,   mcolors.to_rgb('#fc8d59')),
    (0.70,   mcolors.to_rgb('#e31a1c')),
    (1.0,    mcolors.to_rgb('#7f0000')),
])


def get_cmap_norm(feat, absmax):
    if feat in ('heat anom', 'stor discharge anom'):
        return CMAP_HEAT, mcolors.Normalize(vmin=-1, vmax=absmax)
    elif feat == 'nettrans anom':
        return CMAP_OPS, mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    else:
        return CMAP_CF, mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)


def compute_medoids(df_feat, cluster_col, meta_cols):
    excl = set(meta_cols) | {cluster_col}
    feat_cols = [c for c in df_feat.columns if c not in excl]
    X_sc  = StandardScaler().fit_transform(df_feat[feat_cols].values)
    X_pca = PCA(n_components=0.90, random_state=42).fit_transform(X_sc)
    labels = df_feat[cluster_col].values
    k = int(labels.max()) + 1
    medoids = {}
    for c in range(k):
        mask = np.where(labels == c)[0]
        dists = pairwise_distances(X_pca[mask]).sum(axis=1)
        medoids[c] = int(mask[np.argmin(dists)])
    return medoids


def build_scale(df_feat, feat_types, fixed_scale, zone_names):
    scale = {}
    for ft in feat_types:
        if ft in fixed_scale:
            scale[ft] = float(fixed_scale[ft])
        else:
            cols = [f'{ft} {z}' for z in zone_names]
            vals = df_feat[cols].values
            scale[ft] = float(np.ceil(np.percentile(np.abs(vals), 95) / 10) * 10)
    return scale


CONFIGS = {
    'climate': {
        'cluster_col': 'cluster_climate',
        'feat_types':  ['wind anom', 'pv anom', 'heat anom'],
        'feat_labels': {
            'wind anom': 'Wind CF anomaly (%)',
            'pv anom':   'PV CF anomaly (%)',
            'heat anom': 'Heat demand anomaly (%)',
        },
        'fixed_scale': {'wind anom': 100, 'pv anom': 100, 'heat anom': 100},
        'datasets': {
            'future':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/features/features_alpha25_climate_future_clustered.csv'),
            'historical': os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/features/features_alpha25_climate_historical_clustered.csv'),
        },
        'zones': {
            'future':     ZONES_FUTURE,
            'historical': ZONES_HISTORICAL,
        },
    },
    'operational': {
        'cluster_col': 'cluster_operational',
        'feat_types':  ['nettrans anom', 'stor level anom', 'stor discharge anom'],
        'feat_labels': {
            'nettrans anom':       'Net transfer anom. (GWh/h)',
            'stor level anom':     'Storage level anom. (GWh)',
            'stor discharge anom': 'Storage discharge anom. (GWh/h)',
        },
        'fixed_scale': {'stor discharge anom': 10},
        'datasets': {
            'future':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/features/features_alpha25_operational_future_clustered.csv'),
            'historical': os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/features/features_alpha25_operational_historical_clustered.csv'),
        },
        'zones': {
            'future':     ZONES_FUTURE,
            'historical': ZONES_HISTORICAL,
        },
    },
}

# ── Load shapefile, reproject to EPSG:3035 ────────────────────────────────────
gdf_world = gpd.read_file(SHAPEFILE)
europe    = gdf_world.cx[-15:42, 32:73].to_crs('EPSG:3035')

X_MIN, Y_MIN = 2_100_000, 1_300_000
X_MAX, Y_MAX = 6_100_000, 5_700_000

# ── Main: one PNG per (feature × dataset) ────────────────────────────────────
for cl_type, cfg in CONFIGS.items():
    for ds_name in ('future', 'historical'):
        feat_csv = cfg['datasets'][ds_name]
        zones    = cfg['zones'][ds_name]

        if not os.path.exists(feat_csv):
            print(f'SKIP: {feat_csv} not found.')
            continue

        df_feat = pd.read_csv(feat_csv)

        iso_to_zone = {}
        for zone, countries in zones.items():
            for c in countries:
                iso = CODE_MAP.get(c)
                if iso:
                    iso_to_zone[iso] = zone

        meta_cols   = ['sc', 't_start', 't_end']
        cluster_col = cfg['cluster_col']
        zone_names  = list(zones.keys())

        medoids       = compute_medoids(df_feat, cluster_col, meta_cols)
        scale         = build_scale(df_feat, cfg['feat_types'], cfg['fixed_scale'], zone_names)
        cluster_sizes = df_feat[cluster_col].value_counts().sort_index().to_dict()

        for feat in cfg['feat_types']:
            wind     = feat in ('wind anom', 'pv anom')
            heat     = feat == 'heat anom'
            nettrans = feat == 'nettrans anom'
            shared_cb = wind or heat or nettrans or feat == 'stor level anom' or feat == 'stor discharge anom'
            absmax = 80 if wind else scale[feat]
            cmap, norm = get_cmap_norm(feat, absmax)

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            for cl_idx in range(4):
                ax       = axes[cl_idx // 2, cl_idx % 2]
                row_vals = df_feat.iloc[medoids[cl_idx]]
                n_ev     = cluster_sizes.get(cl_idx, 0)

                val_map = {}
                for zone, countries in zones.items():
                    col = f'{feat} {zone}'
                    if col in row_vals.index:
                        v = float(row_vals[col])
                        for c in countries:
                            iso = CODE_MAP.get(c)
                            if iso:
                                val_map[iso] = v

                europe_plot = europe.copy()
                europe_plot['val'] = europe_plot['ADM0_A3'].map(val_map)

                europe_plot[europe_plot['val'].isna()].plot(
                    ax=ax, color='#eeeeee', edgecolor='#aaaaaa', linewidth=0.3,
                )
                modelled = europe_plot[europe_plot['val'].notna()]
                if not modelled.empty:
                    modelled.plot(
                        ax=ax, column='val', cmap=cmap, norm=norm,
                        edgecolor='white', linewidth=0.5,
                    )

                ax.set_xlim(X_MIN, X_MAX)
                ax.set_ylim(Y_MIN, Y_MAX)
                ax.set_aspect('equal')
                ax.axis('off')

                title = f'Cluster {cl_idx+1} (n = {n_ev})' if shared_cb else f'Cluster {cl_idx+1}'
                color = 'black' if shared_cb else CLUSTER_COLORS[cl_idx]
                ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=4)

                if not shared_cb:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cb = plt.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, aspect=22)
                    cb.ax.tick_params(labelsize=8)

            feat_label = cfg['feat_labels'][feat]
            if not shared_cb:
                fig.suptitle(f'{feat_label} — {ds_name}', fontsize=13,
                             fontweight='bold', y=1.01)

            if shared_cb:
                fig.tight_layout()
                # position colorbar just after the rightmost subplot
                positions = [ax.get_position() for ax in axes.ravel()]
                right = max(p.x1 for p in positions)
                bottom = min(p.y0 for p in positions)
                top    = max(p.y1 for p in positions)
                cax = fig.add_axes([right + 0.01, bottom, 0.025, top - bottom])
                sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm2.set_array([])
                cb2 = fig.colorbar(sm2, cax=cax)
                if feat == 'wind anom':
                    cb_label = 'Wind CF anomaly (%)'
                elif feat == 'pv anom':
                    cb_label = 'PV CF anomaly (%)'
                elif feat == 'heat anom':
                    cb_label = 'Heat demand anomaly (%)'
                elif feat == 'nettrans anom':
                    cb_label = 'Net transfer anomaly (GWh/h)'
                elif feat == 'stor level anom':
                    cb_label = 'Storage level anomaly (GWh)'
                else:
                    cb_label = 'Storage rate anomaly (GW)'
                cb2.set_label(cb_label, fontsize=10)
                cb2.ax.tick_params(labelsize=9)
            else:
                fig.tight_layout()

            slug = feat.replace(' ', '_')
            out  = os.path.join(OUT_DIR, ds_name, f'medoid_{slug}_{ds_name}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved: {out}')

print('\nDone.')
