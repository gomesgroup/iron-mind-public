import pandas as pd
import json, os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'SF Pro Display'

dataset_names = [
    'Suzuki_Cernak',
    'amide_coupling_hte',
    'Reductive_Amination',
    'Suzuki_Doyle',
    'Chan_Lam_Full',
    'Buchwald_Hartwig'
]

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

# Subplot positions: maps dataset_name (lowercase) to (row, col) in 2x3 grid
dataset_to_pos = {ds: (i // 3, i % 3) for i, ds in enumerate(dataset_order)}


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


def get_obj_columns(dataset_name):
    config = dataset_to_obj[dataset_name]
    if isinstance(config, str):
        return [config]
    return config['objectives']


def compute_objective(row, dataset_name):
    config = dataset_to_obj[dataset_name]
    if isinstance(config, str):
        return float(row[config])
    objectives = [float(row[obj]) for obj in config['objectives']]
    order = config.get('order', [0, 1])
    desired, undesired = objectives[order[0]], objectives[order[1]]
    total = desired + undesired
    return (desired / total) * desired if total > 0 else 0.0


def match_predictions(predictions_df, original_df, dataset_name):
    """Match predictions to ground truth, return (actuals, predicteds) arrays."""
    obj_cols = get_obj_columns(dataset_name)
    param_cols = [c for c in original_df.columns if c not in obj_cols]
    pred_obj_cols = [c for c in predictions_df.columns if c.endswith('_predicted')]

    actuals, predicteds = [], []
    for _, row in predictions_df.iterrows():
        mask = pd.Series(True, index=original_df.index)
        skip = False
        for col in param_cols:
            if col not in predictions_df.columns:
                continue
            val = str(row[col])
            if val == 'nan' or val == '':
                skip = True
                break
            mask &= original_df[col].astype(str) == val
        if skip:
            continue

        matched = original_df[mask]
        if len(matched) == 0:
            continue

        actual = compute_objective(matched.iloc[0], dataset_name)

        if len(pred_obj_cols) == 1:
            try:
                predicted = float(row[pred_obj_cols[0]])
            except Exception as e:
                predicted = np.nan
                print(f"  ERROR: {e}, setting predicted to NaN")
        else:
            config = dataset_to_obj[dataset_name]
            obj_vals = []
            for obj_name in config['objectives']:
                pcol = f'{obj_name}_predicted'
                if pcol in row:
                    obj_vals.append(float(row[pcol]))
                else:
                    obj_vals = None
                    break
            if obj_vals is None:
                continue
            order = config.get('order', [0, 1])
            d, u = obj_vals[order[0]], obj_vals[order[1]]
            total = d + u
            predicted = (d / total) * d if total > 0 else 0.0

        if np.isnan(actual) or np.isnan(predicted):
            continue

        # Auto-rescale 0-1 predictions to 0-100 if needed
        if actual > 10 and predicted <= 1.5:
            predicted *= 100

        actuals.append(actual)
        predicteds.append(predicted)

    return np.array(actuals), np.array(predicteds)


def style_axes(ax):
    """Apply unified axis styling consistent with iron-mind-public figures."""
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.grid(False, axis='x')
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)


def plot_r2_heatmap(r2_matrix):
    """Create a heatmap of R² (w.r.t. y=x) across all models and datasets."""
    if not r2_matrix:
        print("No R² data to plot heatmap")
        return

    # Datasets on y-axis (rows)
    row_labels = [dataset_display_name.get(ds, ds) for ds in dataset_order]
    row_colors = [dataset_to_color[ds] for ds in dataset_order]

    # Models on x-axis (columns), sorted alphabetically
    model_names = sorted(r2_matrix.keys())

    # Build matrix: rows=datasets, cols=models
    matrix = []
    for ds_key in dataset_order:
        row = []
        for model in model_names:
            row.append(r2_matrix[model].get(ds_key, np.nan))
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.8 + 2), max(4, len(row_labels) * 0.8 + 2)))

    # Diverging colormap centered at 0
    vmin = min(np.nanmin(matrix), -0.1)
    vmax = max(np.nanmax(matrix), 0.1)
    abs_max = max(abs(vmin), abs(vmax))
    cmap = LinearSegmentedColormap.from_list(
        'leakage', ['#d55e00', '#f5f5f5', '#0071b2'], N=256)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-abs_max, vmax=abs_max)

    # X-axis: models (use linebreaks for long names)
    # Strip date suffixes (e.g., -20250514) for cleaner labels
    display_names = [re.sub(r'-\d{8}', '', m) for m in model_names]
    x_labels = [m.replace('/', '\n').replace('-', '-\n', 1) if len(m) > 15 else m for m in display_names]
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(x_labels, fontsize=12, ha='center')

    # Y-axis: datasets with color coding
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=14)

    # Color y-axis tick labels by dataset color (darken light colors for readability)
    for tick_label, color in zip(ax.get_yticklabels(), row_colors):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if luminance > 0.5:
            scale = 0.55
            r, g, b = int(r * scale), int(g * scale), int(b * scale)
            color = f'#{r:02x}{g:02x}{b:02x}'
        tick_label.set_color(color)
        tick_label.set_fontweight('bold')

    # Annotate cells with R² values
    for i in range(len(row_labels)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=13, color='gray')
            else:
                text_color = 'white' if abs(val) > abs_max * 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('$R^2$ (w.r.t. y=x)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(1.5)

    # Styling
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.set_title('Direct Prediction $R^2$ (w.r.t. y=x)', fontsize=18, fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs('./pngs', exist_ok=True)
    plt.savefig('./pngs/figure_8.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure 8 saved to ./pngs/figure_8.png')


def discover_and_plot(run_path):
    """Discover all direct-predict runs and make a 2x3 grid per model (one subplot per dataset)."""
    llm_path = os.path.join(run_path, 'llm')
    if not os.path.exists(llm_path):
        print(f"No llm directory found at {llm_path}")
        return

    # Collect data grouped by model
    # model_data[model_name] = {dataset_name: (actuals, predicteds)}
    model_data = {}

    for dataset_name in sorted(os.listdir(llm_path)):
        benchmark_dir = os.path.join(llm_path, dataset_name, 'benchmark')
        if not os.path.isdir(benchmark_dir):
            continue

        for entry in sorted(os.listdir(benchmark_dir)):
            if 'direct-predict' not in entry:
                continue
            run_dir = os.path.join(benchmark_dir, entry)
            if not os.path.isdir(run_dir):
                continue

            predictions_path = os.path.join(run_dir, 'predictions.csv')
            original_path = os.path.join(run_dir, 'original_data.csv')
            if not os.path.exists(predictions_path) or not os.path.exists(original_path):
                continue

            # Derive model name from directory, normalizing naming inconsistencies
            # (config.json loses the -medium suffix, causing collisions)
            model_name = normalize_model_name(
                re.sub(r'-\d+-direct-predict$', '', entry))

            # Skip Claude 4.6 models (opus, sonnet 4.6)
            if '4-6' in model_name:
                continue
                
            predictions_df = pd.read_csv(predictions_path)
            original_df = pd.read_csv(original_path)
            try:
                actuals, predicteds = match_predictions(predictions_df, original_df, dataset_name)
            except Exception as e:
                raise ValueError(f"Error matching predictions for {model_name}/{dataset_name}: {e}")

            if len(actuals) < 2:
                print(f"  Skipping {model_name}/{dataset_name}: only {len(actuals)} matches")
                continue

            if model_name not in model_data:
                model_data[model_name] = {}
            model_data[model_name][dataset_name] = (actuals, predicteds)

    os.makedirs('./pngs', exist_ok=True)

    # Build R² matrix for heatmap
    r2_matrix = {}  # model_name -> {dataset_key: r2}
    for model_name, datasets in model_data.items():
        r2_matrix[model_name] = {}
        for ds_name, (actuals, predicteds) in datasets.items():
            ss_res = np.sum((actuals - predicteds) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_matrix[model_name][ds_name.lower()] = r2

    # Only include models with data for all datasets in the heatmap
    complete_r2_matrix = {
        model: ds_r2 for model, ds_r2 in r2_matrix.items()
        if all(ds_key in ds_r2 for ds_key in dataset_order)
    }
    plot_r2_heatmap(complete_r2_matrix)

    for model_name, datasets in model_data.items():
        # Skip models that don't have data for all datasets
        ds_keys_available = {ds_name.lower() for ds_name in datasets}
        if not all(ds_key in ds_keys_available for ds_key in dataset_order):
            missing = [ds_key for ds_key in dataset_order if ds_key not in ds_keys_available]
            print(f'Skipping {model_name}: missing datasets {missing}')
            continue

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        for ds_key in dataset_order:
            row, col = dataset_to_pos[ds_key]
            ax = axes[row, col]

            # Find matching dataset_name (case-insensitive)
            matched_ds = None
            for ds_name in datasets:
                if ds_name.lower() == ds_key:
                    matched_ds = ds_name
                    break

            if matched_ds is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(dataset_display_name.get(ds_key, ds_key), fontsize=16, fontweight='bold')
                style_axes(ax)
                continue

            actuals, predicteds = datasets[matched_ds]
            color = dataset_to_color[ds_key]

            ax.scatter(actuals, predicteds, c=color, alpha=0.8, edgecolors='black',
                       linewidths=0.5, s=60, zorder=3)

            # y = x line
            lo = min(actuals.min(), predicteds.min())
            hi = max(actuals.max(), predicteds.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    'k--', alpha=0.5, linewidth=1.5)

            # Stats annotation: R² relative to y=x (not regression line)
            ss_res = np.sum((actuals - predicteds) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ax.text(0.95, 0.05, f'$R^2$={r2:.2f}, n={len(actuals)}',
                    transform=ax.transAxes, fontsize=14,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='k'))

            ax.set_title(dataset_display_name.get(ds_key, matched_ds), fontsize=16, fontweight='bold')
            ax.set_aspect('equal', adjustable='datalim')
            style_axes(ax)

        # Shared axis labels
        for ax in axes[-1, :]:
            ax.set_xlabel('Actual', fontsize=16)
        for ax in axes[:, 0]:
            ax.set_ylabel('Predicted', fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        fig.suptitle(model_name, fontsize=20, fontweight='bold', x=0.5, ha='center')

        safe_name = model_name.replace('/', '_').replace(',', '_')
        fname = f'./pngs/figure_S15_{safe_name}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Figure S15 saved to {fname}')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')

    discover_and_plot(run_path)
