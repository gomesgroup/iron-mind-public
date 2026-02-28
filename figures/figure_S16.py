import pandas as pd
import json, os, sys, glob, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu

plt.rcParams['font.family'] = 'SF Pro Display'

dataset_to_obj = {
    'Buchwald_Hartwig': 'yield',
    'Suzuki_Doyle': 'yield',
    'Suzuki_Cernak': 'conversion',
    'Reductive_Amination': 'percent_conversion',
    'amide_coupling_hte': 'yield',
    'Chan_Lam_Full': {
        'objectives': ['desired_yield', 'undesired_yield'],
        'transform': 'weighted_selectivity',
        'order': [0, 1],
        'aggregation': 'min'
    }
}

dataset_to_color = {
    'buchwald_hartwig': '#000000',
    'chan_lam_full': '#0071b2',
    'suzuki_doyle': '#009e74',
    'reductive_amination': '#cc797f',
    'amide_coupling_hte': '#d55e00',
    'suzuki_cernak': '#f0e142',
}
dataset_order = list(reversed(dataset_to_color.keys()))

dataset_display_name = {
    'buchwald_hartwig': 'Buchwald-Hartwig',
    'chan_lam_full': 'Chan-Lam',
    'suzuki_doyle': 'Suzuki Yield',
    'reductive_amination': 'Reductive Amination',
    'amide_coupling_hte': 'Amide Coupling',
    'suzuki_cernak': 'Suzuki Conversion',
}


def strip_run_config(name):
    """Strip -N-N-N run config suffix (e.g., -1-20-20) from a directory name."""
    return re.sub(r'-\d+-\d+-\d+$', '', name)


def normalize_model_name(name):
    """Normalize model name to handle naming inconsistencies across runs.

    Handles:
      - claude-4-sonnet vs claude-sonnet-4 (word order)
      - -medium vs -thinking (extended thinking suffix)
      - gemini -preview-XX-XX version suffixes
    """
    name = re.sub(r'claude-4-sonnet', 'claude-sonnet-4', name)
    name = name.replace('-medium', '-thinking')
    name = re.sub(r'-preview-\d{2}-\d{2}', '', name)
    return name


def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0
    delta = sum(xi > yi for xi in x for yi in y) - sum(xi < yi for xi in x for yi in y)
    return delta / (n1 * n2)


def get_objective_value(obs, dataset_config):
    """Extract objective value from a single observation."""
    if isinstance(dataset_config, str):
        return float(obs.get(dataset_config, np.nan))
    else:
        objectives = [float(obs.get(obj, np.nan)) for obj in dataset_config['objectives']]
        if dataset_config['transform'] == 'weighted_selectivity':
            order = dataset_config.get('order', [0, 1])
            desired = objectives[order[0]]
            undesired = objectives[order[1]]
            total = desired + undesired
            if total > 0:
                return (desired / total) * desired
            return 0.0
        else:
            raise ValueError(f"Unknown transform: {dataset_config['transform']}")


def lookup_real_yield(obs, original_df, dataset_name):
    """Look up the real (unpermuted) yield for a set of parameters."""
    dataset_config = dataset_to_obj[dataset_name]

    if isinstance(dataset_config, str):
        obj_cols = [dataset_config]
    else:
        obj_cols = dataset_config['objectives']

    param_cols = [c for c in original_df.columns if c not in obj_cols]

    # Build match mask
    mask = pd.Series(True, index=original_df.index)
    for col in param_cols:
        if col in obs:
            mask &= original_df[col].astype(str) == str(obs[col])

    matched = original_df[mask]
    if len(matched) == 0:
        return np.nan

    if isinstance(dataset_config, str):
        return float(matched[dataset_config].iloc[0])
    else:
        if len(matched) > 1:
            vals = []
            for _, row in matched.iterrows():
                objectives = [float(row[obj]) for obj in dataset_config['objectives']]
                order = dataset_config.get('order', [0, 1])
                desired = objectives[order[0]]
                undesired = objectives[order[1]]
                total = desired + undesired
                if total > 0:
                    vals.append((desired / total) * desired)
                else:
                    vals.append(0.0)
            return min(vals)
        else:
            row = matched.iloc[0]
            objectives = [float(row[obj]) for obj in dataset_config['objectives']]
            order = dataset_config.get('order', [0, 1])
            desired = objectives[order[0]]
            undesired = objectives[order[1]]
            total = desired + undesired
            if total > 0:
                return (desired / total) * desired
            return 0.0


def load_permuted_runs(permuted_dir, dataset_name):
    """Load all runs from a permuted-labels directory.

    Returns:
        permuted_tracks: list of cumulative-best arrays (under permuted labels)
        real_tracks: list of cumulative-best arrays (real yields for same suggestions)
        original_df: the original (unpermuted) results DataFrame
    """
    original_csv = os.path.join(permuted_dir, 'original_results.csv')

    if not os.path.exists(original_csv):
        print(f"Warning: {original_csv} not found")
        return None, None, None

    original_df = pd.read_csv(original_csv)
    dataset_config = dataset_to_obj[dataset_name]

    run_dirs = sorted(
        [os.path.join(permuted_dir, d) for d in os.listdir(permuted_dir) if d.startswith('run_')],
        key=lambda x: int(x.split('_')[-1])
    )

    permuted_tracks = []
    real_tracks = []

    for rd in run_dirs:
        seen_path = os.path.join(rd, 'seen_observations.json')
        if not os.path.exists(seen_path):
            continue

        with open(seen_path, 'r') as f:
            seen_data = json.load(f)

        permuted_values = []
        real_values = []

        for obs in seen_data:
            # Permuted objective (what the LLM actually saw)
            pval = get_objective_value(obs, dataset_config)
            permuted_values.append(pval if not np.isnan(pval) else 0)

            # Real objective (what it would have been without permutation)
            rval = lookup_real_yield(obs, original_df, dataset_name)
            real_values.append(rval if not np.isnan(rval) else 0)

        if len(permuted_values) > 0:
            permuted_tracks.append(np.maximum.accumulate(permuted_values))
            real_tracks.append(np.maximum.accumulate(real_values))

    return permuted_tracks, real_tracks, original_df


def load_original_runs(original_dir, dataset_name, n_tracks=20, track_size=20):
    """Load original (non-permuted) optimization runs for comparison."""
    run_dirs = sorted(
        [os.path.join(original_dir, d) for d in os.listdir(original_dir) if d.startswith('run_')],
        key=lambda x: int(x.split('_')[-1])
    )

    dataset_config = dataset_to_obj[dataset_name]
    tracks = []

    for rd in run_dirs[:n_tracks]:
        seen_path = os.path.join(rd, 'seen_observations.json')
        if not os.path.exists(seen_path):
            continue

        with open(seen_path, 'r') as f:
            seen_data = json.load(f)

        values = []
        for obs in seen_data:
            val = get_objective_value(obs, dataset_config)
            values.append(val if not np.isnan(val) else 0)

        if len(values) > 0:
            track = np.maximum.accumulate(values)[:track_size]
            tracks.append(track)

    return tracks


def simulate_random_baseline(original_df, dataset_name, n_runs=20, budget=20):
    """Simulate random baseline performance from the original dataset."""
    dataset_config = dataset_to_obj[dataset_name]
    tracks = []

    if isinstance(dataset_config, str):
        obj_cols = [dataset_config]
    else:
        obj_cols = dataset_config['objectives']

    for _ in range(n_runs):
        sampled = original_df.sample(n=min(budget, len(original_df)), replace=False)
        values = []
        for _, row in sampled.iterrows():
            obs = row.to_dict()
            val = get_objective_value(obs, dataset_config)
            values.append(val if not np.isnan(val) else 0)
        tracks.append(np.maximum.accumulate(values))

    return tracks


def gather_dataset_data(permuted_dir, dataset_name, original_dir=None):
    """Load all permutation data for a single model-dataset combination.

    Returns dict with keys: permuted_tracks, real_tracks, original_tracks,
    random_tracks, permuted_best, real_best, original_best, random_best,
    or None if no data found.
    """
    permuted_tracks, real_tracks, original_df = load_permuted_runs(permuted_dir, dataset_name)

    if permuted_tracks is None or len(permuted_tracks) == 0:
        print(f"No data found for {permuted_dir}")
        return None

    original_tracks = None
    if original_dir is not None and os.path.exists(original_dir):
        original_tracks = load_original_runs(original_dir, dataset_name)

    random_tracks = simulate_random_baseline(original_df, dataset_name)

    return {
        'permuted_tracks': permuted_tracks,
        'real_tracks': real_tracks,
        'original_tracks': original_tracks,
        'random_tracks': random_tracks,
        'permuted_best': [float(np.max(t)) for t in permuted_tracks],
        'real_best': [float(np.max(t)) for t in real_tracks],
        'original_best': [float(np.max(t)) for t in original_tracks] if original_tracks else None,
        'random_best': [float(np.max(t)) for t in random_tracks],
    }


def discover_and_plot(run_path):
    """Auto-discover all permuted-label runs and create delta heatmap."""
    llm_path = os.path.join(run_path, 'llm')
    if not os.path.exists(llm_path):
        print(f"No llm directory found at {llm_path}")
        return

    # Collect data grouped by model
    # model_data[model_name] = {dataset_name: {...}}
    model_data = {}

    for dataset_name in sorted(os.listdir(llm_path)):
        dataset_dir = os.path.join(llm_path, dataset_name, 'benchmark')
        if not os.path.isdir(dataset_dir):
            continue

        for entry in sorted(os.listdir(dataset_dir)):
            if 'permuted-labels' not in entry:
                continue

            permuted_dir = os.path.join(dataset_dir, entry)
            if not os.path.isdir(permuted_dir):
                continue

            # Clean model name: strip permuted-labels tag, run config suffix,
            # then normalize naming inconsistencies
            # e.g. "claude-4-sonnet-20250514-medium-1-20-20-permuted-labels"
            #    -> "claude-sonnet-4-20250514-thinking"
            model_name = normalize_model_name(
                strip_run_config(entry.replace('-permuted-labels', '')))

            # Match to original (non-permuted) run using normalized names
            original_dir = None
            for bm_entry in os.listdir(dataset_dir):
                if 'permuted' in bm_entry or 'direct-predict' in bm_entry:
                    continue
                candidate = os.path.join(dataset_dir, bm_entry)
                if not os.path.isdir(candidate):
                    continue
                orig_name = normalize_model_name(strip_run_config(bm_entry))
                if orig_name == model_name:
                    original_dir = candidate
                    break

            print(f'Loading: {model_name} / {dataset_name}')
            data = gather_dataset_data(permuted_dir, dataset_name, original_dir)
            if data is not None:
                if model_name not in model_data:
                    model_data[model_name] = {}
                model_data[model_name][dataset_name] = data

    # Build delta matrix (only models with all datasets)
    delta_matrix = {}

    for model_name, datasets_data in model_data.items():
        ds_keys_available = {ds_name.lower() for ds_name in datasets_data}
        if not all(ds_key in ds_keys_available for ds_key in dataset_order):
            missing = [ds_key for ds_key in dataset_order if ds_key not in ds_keys_available]
            print(f'Skipping {model_name}: missing datasets {missing}')
            continue

        delta_matrix[model_name] = {}
        for ds_name, data in datasets_data.items():
            ds_key = ds_name.lower()
            d_orig = cliffs_delta(data['original_best'], data['random_best']) if data['original_best'] else np.nan
            d_perm = cliffs_delta(data['real_best'], data['random_best'])
            delta_matrix[model_name][ds_key] = d_orig - d_perm

    # Plot heatmap
    if not delta_matrix:
        print("No delta data to plot heatmap")
        return

    row_labels = [dataset_display_name.get(ds, ds) for ds in dataset_order]
    row_colors = [dataset_to_color[ds] for ds in dataset_order]
    model_names = sorted(delta_matrix.keys())

    matrix = []
    for ds_key in dataset_order:
        row = []
        for model in model_names:
            row.append(delta_matrix[model].get(ds_key, np.nan))
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(
        figsize=(max(10, len(model_names) * 1.8 + 2),
                 max(4, len(row_labels) * 0.8 + 2)))

    vmin = min(np.nanmin(matrix), -0.1)
    vmax = max(np.nanmax(matrix), 0.1)
    abs_max = max(abs(vmin), abs(vmax))
    cmap = LinearSegmentedColormap.from_list(
        'leakage', ['#d55e00', '#f5f5f5', '#0071b2'], N=256)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-abs_max, vmax=abs_max)

    # Strip date suffixes (e.g., -20250514) for cleaner labels
    display_names = [re.sub(r'-\d{8}', '', m) for m in model_names]
    x_labels = [m.replace('/', '\n').replace('-', '-\n', 1) if len(m) > 15 else m
                for m in display_names]
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(x_labels, fontsize=12, ha='center')

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=14)

    for tick_label, color in zip(ax.get_yticklabels(), row_colors):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if luminance > 0.5:
            scale = 0.55
            r, g, b = int(r * scale), int(g * scale), int(b * scale)
            color = f'#{r:02x}{g:02x}{b:02x}'
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')

    for i in range(len(row_labels)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=13, color='gray')
            else:
                text_color = 'white' if abs(val) > abs_max * 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('δ(orig, rand) − δ(real_perm, rand)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(1.5)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.set_title('Permutation Label Leakage: Cliff\'s Δ Drop', fontsize=18,
                 fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs('./pngs', exist_ok=True)
    plt.savefig('./pngs/figure_S16.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure S16 saved to ./pngs/figure_S16.png')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')

    discover_and_plot(run_path)
