"""
Event Calendar Comparison — Before vs After Filtering
======================================================
Generates two PNG figures (future and historical) showing the stress event
calendar at α = 25%, with and without the quality filters applied.

Each figure has two panels:
  (a) all events before filtering (gray = filtered out, colors = cost share)
  (b) events after applying cost_share ≥ 0.2% AND duration ≥ 1 day

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha25.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv

Outputs:
  htmls/fotos/future/filter_comparison_future_alpha25.png
  htmls/fotos/historical/filter_comparison_historical_alpha25.png
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: PNG figures only
FUT_CSV  = '~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha25.csv'
HIST_CSV = '~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv'
OUT_DIR  = '~/Desktop/Bachelor Thesis/analysis/fotos/'
COST_MIN = 0.2   # minimum cost_share_cross (%) to pass filter
DUR_MIN  = 1.0   # minimum duration (days) to pass filter
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

FUT_CSV  = os.path.expanduser(FUT_CSV)
HIST_CSV = os.path.expanduser(HIST_CSV)
OUT_DIR  = os.path.expanduser(OUT_DIR)

N_SC   = 60
N_DAYS = 365

MS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

SS0, SS1 = 151, 243   # spring/summer band (0-indexed days 152–243)

# ── Load & thresholds ─────────────────────────────────────────────────────────
df_fut  = pd.read_csv(FUT_CSV)
df_hist = pd.read_csv(HIST_CSV)

all_cs = pd.concat([df_fut['cost_share_cross'], df_hist['cost_share_cross']]).values
T1 = np.percentile(all_cs, 40)
T2 = np.percentile(all_cs, 70)
T3 = np.percentile(all_cs, 90)
print(f'Thresholds  T1={T1:.4f}%  T2={T2:.4f}%  T3={T3:.4f}%')

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = np.array([
    [1.000, 1.000, 1.000, 1.00],   # 0 – no event (white)
    [0.992, 0.792, 0.420, 1.00],   # 1 – #FDCA6B  low
    [0.941, 0.549, 0.000, 1.00],   # 2 – #F08C00  mid
    [0.788, 0.290, 0.000, 1.00],   # 3 – #C94A00  high
    [0.490, 0.000, 0.000, 1.00],   # 4 – #7D0000  top
    [0.860, 0.860, 0.860, 0.80],   # 5 – filtered out (light gray)
])

def build_image(df, apply_filter):
    df2 = df.copy()
    df2['d1'] = (df2['t_start'] // 24).clip(0, N_DAYS - 1).astype(int)
    df2['d2'] = (df2['t_end']   // 24).clip(0, N_DAYS - 1).astype(int)
    df2['passes'] = True
    if apply_filter:
        df2['passes'] = (df2['cost_share_cross'] >= COST_MIN) & \
                        (df2['duration_days']    >= DUR_MIN)

    cs_pass = np.zeros((N_SC, N_DAYS))
    cs_fail = np.zeros((N_SC, N_DAYS))
    for _, row in df2.iterrows():
        sc = int(row['sc']) - 1
        d1, d2 = int(row['d1']), int(row['d2'])
        cs = float(row['cost_share_cross'])
        if row['passes']:
            cs_pass[sc, d1:d2 + 1] = np.maximum(cs_pass[sc, d1:d2 + 1], cs)
        else:
            cs_fail[sc, d1:d2 + 1] = np.maximum(cs_fail[sc, d1:d2 + 1], cs)

    cidx = np.zeros((N_SC, N_DAYS), dtype=int)
    cidx[(cs_fail > 0) & (cs_pass == 0)]   = 5
    cidx[(cs_pass > 0) & (cs_pass < T1)]   = 1
    cidx[(cs_pass >= T1) & (cs_pass < T2)] = 2
    cidx[(cs_pass >= T2) & (cs_pass < T3)] = 3
    cidx[cs_pass >= T3]                    = 4
    return PALETTE[cidx]

def event_stats(df, apply_filter):
    sub = df[(df['cost_share_cross'] >= COST_MIN) & (df['duration_days'] >= DUR_MIN)] \
          if apply_filter else df
    return len(sub), sub['sc'].nunique(), sub['duration_days'].mean() if len(sub) else 0

# ── Per-dataset figure ─────────────────────────────────────────────────────────
DATASETS = [
    (df_fut,  'Future',     'future/filter_comparison_future_alpha25.png'),
    (df_hist, 'Historical', 'historical/filter_comparison_historical_alpha25.png'),
]

PANEL_LABELS = ['(a)', '(b)']
PANEL_TITLES = ['all events', 'filtered events']

for df, ds_label, fname in DATASETS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={'wspace': 0.08})
    fig.patch.set_facecolor('white')

    for col_i, (af, ptitle, plabel) in \
            enumerate(zip([False, True], PANEL_TITLES, PANEL_LABELS)):
        ax = axes[col_i]
        ax.set_facecolor('white')

        img = build_image(df, af)
        n_ev, n_sc, avg_dur = event_stats(df, af)

        # Calendar image
        ax.imshow(img, aspect='auto', interpolation='nearest',
                  extent=[-0.5, N_DAYS - 0.5, N_SC - 0.5, -0.5])

        # Spring/summer band
        ax.axvspan(SS0 - 0.5, SS1 - 0.5,
                   color='#3B6D11', alpha=0.06, zorder=2, lw=0)

        # Month grid lines
        for m in MS[1:]:
            ax.axvline(m - 1.5, color='#CCCCCC', linewidth=0.4, zorder=3)

        # Axes limits & ticks
        ax.set_xlim(-0.5, N_DAYS - 0.5)
        ax.set_ylim(N_SC - 0.5, -0.5)

        x_ticks, x_labels = [], []
        for i, m_start in enumerate(MS):
            m_end = MS[i + 1] if i < 11 else 366
            x_ticks.append((m_start - 1 + m_end - 2) / 2)
            x_labels.append(MN[i])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9, color='#333333')

        ax.set_yticks([i - 0.5 for i in range(10, 61, 10)])
        ax.set_yticklabels(
            [f's{i}' for i in range(10, 61, 10)],
            fontsize=8, color='#555555'
        )
        if col_i == 1:
            ax.set_yticklabels([])

        for sp in ax.spines.values():
            sp.set_color('#BBBBBB')
            sp.set_linewidth(0.6)
        ax.tick_params(length=0)

        # Panel label + title
        ax.set_title(f'{plabel}  {ptitle}', fontsize=11,
                     loc='left', pad=6, color='#1A1A18', fontweight='normal')

        # Event count annotation (top-right corner, subtle)
        ax.text(0.99, 1.01,
                f'{n_ev} events  ·  {n_sc}/60 scenarios  ·  avg {avg_dur:.1f} d',
                transform=ax.transAxes, fontsize=8,
                color='#888888', va='bottom', ha='right')

    # Shared y-label
    axes[0].set_ylabel('scenario', fontsize=9, color='#555555', labelpad=6)

    # Figure title
    fig.suptitle(f'{ds_label} dataset  ·  α = 25%',
                 fontsize=12, y=1.01, color='#1A1A18', fontweight='normal',
                 x=0.5, ha='center')

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#FDCA6B', edgecolor='#CCCCCC', linewidth=0.4,
                       label=f'< {T1:.3f}%'),
        mpatches.Patch(facecolor='#F08C00', edgecolor='none',
                       label=f'{T1:.3f}–{T2:.3f}%'),
        mpatches.Patch(facecolor='#C94A00', edgecolor='none',
                       label=f'{T2:.3f}–{T3:.3f}%'),
        mpatches.Patch(facecolor='#7D0000', edgecolor='none',
                       label=f'≥ {T3:.3f}%'),
        mpatches.Patch(facecolor='#3B6D11', alpha=0.3, edgecolor='none',
                       label='spring / summer'),
        mpatches.Patch(facecolor='#CCCCCC', edgecolor='none',
                       label='filtered out'),
    ]
    fig.legend(
        handles=legend_handles,
        title='cost share (cross-scenario):',
        title_fontsize=8,
        loc='lower center',
        ncol=6,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    out_path = os.path.join(OUT_DIR, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f'Saved: {out_path}')
