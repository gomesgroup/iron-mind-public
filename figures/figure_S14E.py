#!/usr/bin/env python3
"""
Figure S17: Chan-Lam single-objective benchmark comparison.

2-row figure (one row per dataset variant), 3 boxes per method showing
min / mean / max replicate-aggregation strategies side by side.

  Top row    – Chan_Lam_Desired    : objective = desired_yield
  Bottom row – Chan_Lam_Selectivity: objective = weighted_selectivity
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'SF Pro Display'

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
dataset_names = ['Chan_Lam_Desired', 'Chan_Lam_Selectivity']

dataset_to_obj = {
    'Chan_Lam_Desired':     'desired_yield',
    'Chan_Lam_Selectivity': 'weighted_selectivity',
}

# One colour per dataset panel (matches Chan-Lam palette in other figures)
dataset_to_color = {
    'chan_lam_desired':     '#0071b2',   # blue
    'chan_lam_selectivity': '#e69f00',   # orange
}

dataset_display_name = {
    'chan_lam_desired':     'Desired Yield (%)',
    'chan_lam_selectivity': 'Weighted Selectivity (%)',
}

# Aggregation strategies and their display colours / labels
AGGREGATIONS = ['min', 'mean', 'max']
AGG_COLORS   = {
    'min':  '#2c7bb6',   # dark blue  – worst case
    'mean': '#abd9e9',   # light blue – average
    'max':  '#d7191c',   # red        – best case
}
AGG_LABELS = {
    'min':  'Min (worst case)',
    'mean': 'Mean',
    'max':  'Max (best case)',
}

# Method ordering for x-axis (LLM first, then BO)
LLM_ORDER = ['claude-sonnet-4', 'gpt-5']
BO_ORDER   = ['atlas-ei', 'atlas-ei-des', 'atlas-pi', 'atlas-pi-des',
               'atlas-ucb', 'atlas-ucb-des']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_method_name(folder_name: str) -> str:
    """Strip run-config suffixes and normalise model name."""
    name = folder_name
    name = name.replace('-1-20-20', '').replace('-latest', '')
    name = name.replace('-20250514', '').replace('-preview-04-17', '')
    name = name.replace('-preview-06-17', '').replace('-20250805', '')
    name = name.replace('-des0', '-des')
    return name


_AGG_FN = {'min': np.min, 'mean': np.mean, 'max': np.max}


def get_tracks(path: str, dataset_name: str, aggregation: str = 'min',
               bo: bool = False, n_tracks: int = 20, track_size: int = 20):
    """
    Extract cumulative-best optimisation tracks.

    Multiple replicates at the same parameter configuration are collapsed
    to a single value using `aggregation` ('min', 'mean', or 'max').
    """
    run_dirs = sorted(
        [os.path.join(path, d) for d in os.listdir(path) if d.startswith('run_')],
        key=lambda x: int(x.split('_')[-1]),
    )

    agg_fn    = _AGG_FN[aggregation]
    objective = dataset_to_obj[dataset_name]
    exclude_keys = {objective, 'reasoning', 'explanation'}
    tracks = []

    for rd in run_dirs:
        seen_path = os.path.join(rd, 'seen_observations.json')
        if not os.path.exists(seen_path):
            continue

        with open(seen_path) as f:
            seen_data = json.load(f)

        df = pd.DataFrame(seen_data)
        objective_values = []

        if 'reasoning' in df.columns and df['reasoning'].notna().any():
            for _, group in df.groupby('reasoning', sort=False):
                vals = group[objective].dropna().values
                if len(vals) > 0:
                    objective_values.append(float(agg_fn(vals)))
        else:
            param_cols = [c for c in df.columns if c not in exclude_keys]
            if not param_cols:
                for obs in seen_data:
                    v = obs.get(objective)
                    if v is not None and not np.isnan(float(v)):
                        objective_values.append(float(v))
            else:
                for _, group in df.groupby(param_cols, sort=False):
                    vals = group[objective].dropna().values
                    if len(vals) > 0:
                        objective_values.append(float(agg_fn(vals)))

        arr = np.array(objective_values, dtype=float)
        if bo:
            arr = arr[1:]
        arr = arr[:track_size]
        if len(arr) > 0:
            tracks.append(np.maximum.accumulate(arr))

    if len(tracks) < n_tracks:
        return None
    return tracks[:n_tracks]


def top_obs_from_tracks(tracks):
    top_obs = [float(np.max(t)) for t in tracks]
    return {
        'top_obs': top_obs,
        'median':  float(np.median(top_obs)),
        'q1':      float(np.percentile(top_obs, 25)),
        'q3':      float(np.percentile(top_obs, 75)),
    }


def get_top_obs_data(path_dict: dict) -> dict:
    """
    Returns top_obs_data[dataset_key][method_path][aggregation] = stats dict.
    """
    top_obs_data = {}
    for dataset_name, methods in path_dict.items():
        key = dataset_name.lower()
        top_obs_data[key] = {}
        for method_path, agg_tracks in methods.items():
            top_obs_data[key][method_path] = {}
            for agg, tracks in agg_tracks.items():
                if tracks is not None:
                    top_obs_data[key][method_path][agg] = top_obs_from_tracks(tracks)
    return top_obs_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_boxplot(bp, color, iqr):
    for box in bp['boxes']:
        box.set(color='black', linewidth=1.2, zorder=3, facecolor=color, alpha=1.0)
    for w in bp['whiskers']:
        w.set(color='black', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.2)
    for med in bp['medians']:
        med.set(color='white' if iqr > 0 else 'black', linewidth=2.5, zorder=4)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor=color, markeredgecolor='black',
                  markersize=6, alpha=0.9)


def _draw_panel(ax, dataset_key, top_obs_data, method_order, llm_methods, bo_methods):
    """
    Draw one panel. For each method, plot 3 side-by-side boxes:
    min / mean / max replicate aggregation.
    """
    ds_data   = top_obs_data.get(dataset_key, {})
    n_agg     = len(AGGREGATIONS)
    box_width = 0.35
    agg_gap   = 0.42   # gap between aggregation boxes within one method
    method_spacing = n_agg * agg_gap + 1.5   # total spacing between method centres

    method_positions = {name: idx * method_spacing + 1
                        for idx, name in enumerate(method_order)}

    for method_name in method_order:
        base_pos = method_positions[method_name]

        for method_path, agg_data in ds_data.items():
            if normalize_method_name(method_path.split('/')[-1]) != method_name:
                continue

            for i, agg in enumerate(AGGREGATIONS):
                if agg not in agg_data:
                    continue
                stats  = agg_data[agg]
                pos    = base_pos + (i - (n_agg - 1) / 2) * agg_gap
                color  = AGG_COLORS[agg]

                bp = ax.boxplot(
                    stats['top_obs'],
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True,
                    zorder=3,
                )
                _style_boxplot(bp, color, iqr=stats['q3'] - stats['q1'])
            break

    # Thin dashed separator between LLM and BO groups
    if llm_methods and bo_methods:
        sep_x = (method_positions[llm_methods[-1]] + method_positions[bo_methods[0]]) / 2
        ax.axvline(x=sep_x, color='gray', linestyle='--', linewidth=1.2,
                   alpha=0.6, zorder=0)

    positions = [method_positions[m] for m in method_order]
    ax.set_xticks(positions)
    ax.set_xticklabels(method_order, rotation=0, ha='center', fontsize=17)
    ax.set_ylabel(dataset_display_name[dataset_key], fontsize=19)
    ax.set_ylim(0, 105)
    ax.set_xlim(min(positions) - 1.5, max(positions) + 1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.grid(False, axis='x')
    ax.tick_params(axis='y', labelsize=17)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_figure(top_obs_data: dict):
    """2-row × 1-col wide figure; 3 boxes per method (min/mean/max)."""
    from matplotlib.patches import Patch

    def panel_methods(dataset_key):
        ds_data = top_obs_data.get(dataset_key, {})
        present = {normalize_method_name(mp.split('/')[-1]) for mp in ds_data}
        llm = [m for m in LLM_ORDER if m in present]
        bo  = [m for m in BO_ORDER  if m in present]
        return llm, bo

    datasets = ['chan_lam_desired', 'chan_lam_selectivity']
    fig, axes = plt.subplots(2, 1, figsize=(22, 10))
    fig.subplots_adjust(hspace=0.15, top=0.96, bottom=0.12)

    for ax, dataset_key in zip(axes, datasets):
        llm_m, bo_m = panel_methods(dataset_key)
        method_order = llm_m + bo_m
        if not method_order:
            ax.axis('off')
            continue
        _draw_panel(ax, dataset_key, top_obs_data, method_order, llm_m, bo_m)

    # Legend below both panels
    legend_handles = [
        Patch(facecolor=AGG_COLORS[agg], edgecolor='black', label=AGG_LABELS[agg])
        for agg in AGGREGATIONS
    ]
    fig.legend(handles=legend_handles, fontsize=18, loc='lower center',
               ncol=len(AGGREGATIONS), bbox_to_anchor=(0.5, 0.01),
               framealpha=0.9, frameon=True)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f'Using run path: {run_path}')
    else:
        run_path = input('Enter the run path: ')

    n_tracks   = 20
    track_size = 20

    # Build path_dict: path_dict[dataset][method_path][aggregation] = tracks
    path_dict = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        bo_path  = os.path.join(run_path, f'bayesian/{dataset_name}/benchmark/')
        llm_path = os.path.join(run_path, f'llm/{dataset_name}/benchmark/')

        if os.path.exists(bo_path):
            for folder in os.listdir(bo_path):
                full_path = os.path.join(bo_path, folder)
                if not os.path.isdir(full_path):
                    continue
                if folder.endswith('-20') or folder.endswith('-20-des0'):
                    agg_tracks = {
                        agg: get_tracks(full_path, dataset_name, aggregation=agg,
                                        bo=True, n_tracks=n_tracks, track_size=track_size)
                        for agg in AGGREGATIONS
                    }
                    if any(v is not None for v in agg_tracks.values()):
                        path_dict[dataset_name][full_path] = agg_tracks
                    else:
                        print(f'  [skip] insufficient runs: {folder}')

        if os.path.exists(llm_path):
            for folder in os.listdir(llm_path):
                full_path = os.path.join(llm_path, folder)
                if not os.path.isdir(full_path):
                    continue
                if '1-20-20' in folder:
                    agg_tracks = {
                        agg: get_tracks(full_path, dataset_name, aggregation=agg,
                                        bo=False, n_tracks=n_tracks, track_size=track_size)
                        for agg in AGGREGATIONS
                    }
                    if any(v is not None for v in agg_tracks.values()):
                        path_dict[dataset_name][full_path] = agg_tracks
                    else:
                        print(f'  [skip] insufficient runs: {folder}')

    # Report coverage
    for ds, methods in path_dict.items():
        print(f'{ds}: {len(methods)} methods')
        for mp in methods:
            print(f'  {normalize_method_name(mp.split("/")[-1])}')

    top_obs_data = get_top_obs_data(path_dict)

    fig = make_figure(top_obs_data)

    os.makedirs('./pngs', exist_ok=True)
    out_path = './pngs/figure_S14E_Chan_Lam_single_objective.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Figure S14E saved to {out_path}')
