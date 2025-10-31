import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob, os
import sys
from olympus.datasets.dataset import Dataset

# Set font to sf pro
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
        'transform': 'weighted_selectivity',  # (desired/(desired + undesired)) * desired
        'order': [0, 1],  # desired, undesired
        'aggregation': 'min'
    }
}

model_to_provider = {
    "Anthropic": [
        # 'claude-3-5-haiku',
        # 'claude-3-5-sonnet',
        'claude-3-7-sonnet',
        'claude-3-7-sonnet-thinking',
        'claude-sonnet-4',
        'claude-sonnet-4-thinking',
        'claude-opus-4',
        'claude-opus-4-1'
    ],
    "Google": [
        'gemini-2.0-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.5-flash',
        'gemini-2.5-flash-medium',
        'gemini-2.5-pro',
        'gemini-2.5-pro-medium',
    ],
    "OpenAI": [
        'gpt-5-mini',
        'gpt-5',
        'o4-mini-low',
        'o3-low',
        'o4-mini-high',
        'o3-high',
    ],
    "Atlas": [
        'atlas-ei',
        'atlas-ei-des',
        'atlas-ucb',
        'atlas-ucb-des',
        'atlas-pi',
        'atlas-pi-des',
    ]
}

dataset_to_color = {
    'buchwald_hartwig': '#000000',  # Dark blue (darkest)
    'chan_lam_full': '#0071b2',     # Dark Orange
    'suzuki_doyle': '#009e74',      # Light Blue
    'reductive_amination': '#cc797f', # Orange
    'amide_coupling_hte': '#d55e00', # Yellow  
    'suzuki_cernak': '#f0e142',      # Lighter gray
}
dataset_order = list(dataset_to_color.keys())

def compute_individual_run_cumulative_entropy(run_selections, param_options):
    """
    Compute cumulative entropy for a single run across all iterations.
    
    Args:
        run_selections: List of parameter selections for one run (length T=20)
        param_options: Dictionary with parameter names as keys and list of options as values
    
    Returns:
        Single cumulative entropy value for this run
    """
    if not run_selections:
        return 0.0
    
    # Count parameter selections across all iterations in this run
    param_counts = {}
    for param_name, options in param_options.items():
        param_counts[param_name] = defaultdict(int)
    
    # Count selections across all iterations
    total_iterations = len(run_selections)
    for selection in run_selections:
        for param_name in param_options.keys():
            if param_name in selection:
                param_value = selection[param_name]
                if param_value in param_options[param_name]:
                    param_counts[param_name][param_value] += 1
    
    # Compute normalized Shannon entropy for each parameter
    param_entropies = []
    for param_name, options in param_options.items():
        counts = param_counts[param_name]
        
        # Convert counts to probabilities
        probabilities = []
        for option in options:
            prob = counts[option] / total_iterations if total_iterations > 0 else 0
            probabilities.append(prob)
        
        # Compute normalized Shannon entropy
        entropy = 0.0
        n_options = len(options)
        max_entropy = np.log2(n_options) if n_options > 1 else 1
        
        for prob in probabilities:
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        param_entropies.append(normalized_entropy)
    
    # Average entropy across all parameters
    cumulative_entropy = np.mean(param_entropies) if param_entropies else 0.0
    return float(cumulative_entropy)

def compute_individual_run_parameter_entropies(run_selections, param_options):
    """
    Compute cumulative entropy for each parameter separately for a single run.
    
    Args:
        run_selections: List of parameter selections for one run (length T=20)
        param_options: Dictionary with parameter names as keys and list of options as values
    
    Returns:
        Dictionary with parameter names as keys and their cumulative entropies as values
    """
    if not run_selections:
        return {param: 0.0 for param in param_options.keys()}
    
    # Count parameter selections across all iterations in this run
    param_counts = {}
    for param_name, options in param_options.items():
        param_counts[param_name] = defaultdict(int)
    
    # Count selections across all iterations
    total_iterations = len(run_selections)
    for selection in run_selections:
        for param_name in param_options.keys():
            if param_name in selection:
                param_value = selection[param_name]
                if param_value in param_options[param_name]:
                    param_counts[param_name][param_value] += 1
    
    # Compute normalized Shannon entropy for each parameter
    param_entropies = {}
    for param_name, options in param_options.items():
        counts = param_counts[param_name]
        
        # Convert counts to probabilities
        probabilities = []
        for option in options:
            prob = counts[option] / total_iterations if total_iterations > 0 else 0
            probabilities.append(prob)
        
        # Compute normalized Shannon entropy
        entropy = 0.0
        n_options = len(options)
        max_entropy = np.log2(n_options) if n_options > 1 else 1
        
        for prob in probabilities:
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        param_entropies[param_name] = float(normalized_entropy)
    
    return param_entropies

def extract_cumulative_entropy_data(path_dict, dataset_param_options):
    """
    Extract cumulative entropy data similar to how get_top_obs_data works.
    Returns one cumulative entropy value per run per method-dataset combination.
    """
    entropy_data = {}
    
    for dataset_name, optimization_data in path_dict.items():
        entropy_data[dataset_name.lower()] = {}
        param_options = dataset_param_options[dataset_name]
        objectives = dataset_to_obj[dataset_name]
        
        if isinstance(objectives, dict):
            objectives = objectives["objectives"]
        else:
            objectives = [objectives]
        
        for method_path, _ in optimization_data.items():
            entropy_data[dataset_name.lower()][method_path] = {
                'cumulative_entropies': [], 
                'median': None, 
                'q1': None, 
                'q3': None
            }
            
            # Get all run directories for this method
            runs = glob.glob(method_path + "/run_*/")
            run_entropies = []
            
            for run_dir in runs:
                try:
                    # Load seen observations for this run
                    with open(os.path.join(run_dir, "seen_observations.json"), "r") as f:
                        seen_observations = json.load(f)
                    
                    df = pd.DataFrame(seen_observations)
                    
                    # Group observations similar to figure_7.py approach
                    if "llm" in method_path:
                        group_cols = ["reasoning", "explanation"]
                    elif "bayesian" in method_path:
                        all_cols = df.columns.tolist()
                        group_cols = [c for c in all_cols if c not in objectives]
                    
                    # Extract parameter selections for this run
                    grouped_list = list(df.groupby(group_cols, sort=False))
                    run_selections = []
                    
                    for group in grouped_list:
                        group_df = group[1]
                        selection_dict = {}
                        for param_name in param_options.keys():
                            if param_name in group_df.columns:
                                all_values = group_df[param_name].tolist()
                                selection_dict[param_name] = all_values[0]
                        run_selections.append(selection_dict)
                    
                    # Skip first selection for Bayesian methods (like in original code)
                    if "bayesian" in method_path:
                        run_selections = run_selections[1:]
                    
                    # Compute cumulative entropy for this run
                    run_entropy = compute_individual_run_cumulative_entropy(run_selections, param_options)
                    run_entropies.append(run_entropy)
                    
                except Exception as e:
                    print(f'Error processing {run_dir}: {e}')
                    continue
            
            # Store entropy data for this method
            if run_entropies:
                entropy_data[dataset_name.lower()][method_path]['cumulative_entropies'] = run_entropies
                entropy_data[dataset_name.lower()][method_path]['median'] = float(np.median(run_entropies))
                entropy_data[dataset_name.lower()][method_path]['q1'] = float(np.percentile(run_entropies, 25))
                entropy_data[dataset_name.lower()][method_path]['q3'] = float(np.percentile(run_entropies, 75))
            
            print(f"Processed {dataset_name} - {method_path.split('/')[-1]}: {len(run_entropies)} runs")
    
    return entropy_data

def create_individual_provider_entropy_plot(provider_name, provider_method_list, entropy_data, remove_datasets=[]):
    """Create entropy boxplots for one provider, similar to figure_5.py style"""
    if not provider_method_list:
        print(f"No methods found for {provider_name}")
        return None, None, None
    
    entropy_data_refined = {k: v for k, v in entropy_data.items() if k not in remove_datasets}
    
    # Fixed figure sizing consistent with figure_5.py
    fig_width = 15
    fig_height = 8
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Add horizontal line at y=0 in the background
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    
    # Create mapping of method names to their positions with better spacing
    method_positions = {name: idx * 3 + 1 for idx, name in enumerate(provider_method_list)}
    
    provider_data = {}
    for method_name in provider_method_list:
        position = method_positions[method_name]
        offset = 0
        
        for dataset_name, dataset_data in entropy_data_refined.items():
            if dataset_name not in provider_data:
                provider_data[dataset_name] = {}
            
            for key, method_data in dataset_data.items():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '')
                current_method = current_method.replace('-latest', '')
                current_method = current_method.replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '')
                current_method = current_method.replace('-preview-04-17', '')
                current_method = current_method.replace('-des0', '-des')
                current_method = current_method.replace('-preview-06-17', '')
                current_method = current_method.replace('-20250805', '')
                
                if current_method == method_name and method_data['cumulative_entropies']:
                    provider_data[dataset_name][method_name] = method_data['cumulative_entropies']
                    boxplot = ax.boxplot(
                        method_data['cumulative_entropies'],
                        positions=[position + offset],
                        widths=0.3,
                        patch_artist=True,
                        showfliers=True,
                        zorder=3
                    )
                    IQR = np.percentile(method_data['cumulative_entropies'], 75) - np.percentile(method_data['cumulative_entropies'], 25)
                    
                    # Customize boxplot appearance (similar to figure_5.py)
                    for median in boxplot['medians']:
                        if IQR == 0:
                            median.set(color='k', linewidth=4, zorder=2)
                        else:
                            median.set(color='white', linewidth=4, zorder=2)
                    
                    for box in boxplot['boxes']:
                        box.set(color='black', linewidth=2, zorder=3)
                        box.set(facecolor=dataset_to_color[dataset_name.lower()], alpha=0.8, zorder=1)
                    
                    for whisker in boxplot['whiskers']:
                        whisker.set(color='black', linewidth=2, zorder=1)
                    
                    for cap in boxplot['caps']:
                        cap.set(color='black', linewidth=2, zorder=1)
                    
                    for flier in boxplot['fliers']:
                        flier.set(marker='o', markerfacecolor=dataset_to_color[dataset_name.lower()],
                                markeredgecolor='black', markersize=8, alpha=0.8)
                    
                    offset += 0.45
    
    # Calculate centered tick positions (similar to figure_5.py)
    centered_positions = []
    for method_name in provider_method_list:
        base_position = method_positions[method_name]
        # Count how many datasets have data for this method
        n_datasets = 0
        for dataset_name, dataset_data in entropy_data_refined.items():
            for key in dataset_data.keys():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '')
                current_method = current_method.replace('-latest', '')
                current_method = current_method.replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '')
                current_method = current_method.replace('-preview-04-17', '')
                current_method = current_method.replace('-des0', '-des')
                current_method = current_method.replace('-preview-06-17', '')
                current_method = current_method.replace('-20250805', '')
                if current_method == method_name:
                    n_datasets += 1
                    break
        
        # Calculate center position
        center_offset = (n_datasets - 1) * 0.45 / 2
        centered_positions.append(base_position + center_offset)
    
    # Add vertical separator lines between method groups
    if len(provider_method_list) > 1:
        for i in range(len(provider_method_list) - 1):
            separator_x = (centered_positions[i] + centered_positions[i + 1]) / 2
            ax.axvline(x=separator_x, color='gray', linestyle='-', linewidth=2, alpha=0.7, zorder=0)
    
    # Configure the plot
    ax.set_xticks(centered_positions)
    
    # Handle method names with special formatting
    xticklabels = []
    for method in provider_method_list:
        if '-thinking' in method or '-medium' in method:
            xticklabels.append("-".join(method.split('-')[:-1]) + "\n" + method.split('-')[-1])
        else:
            xticklabels.append(method)
    
    ax.set_xticklabels(xticklabels, rotation=0, ha='center', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.grid(False, axis='x')
    ax.set_ylim(0, 1.05)  # Entropy is normalized 0-1
    ax.set_ylabel('Cumulative Entropy', fontsize=16)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    
    # Make plot border black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    # plt.show()
    
    return fig, ax, provider_data

def extract_parameter_specific_entropy_data(path_dict, dataset_param_options):
    """
    Extract parameter-specific cumulative entropy data for SI analysis.
    Returns entropy values for each parameter separately across all runs.
    """
    parameter_entropy_data = {}
    
    for dataset_name, optimization_data in path_dict.items():
        parameter_entropy_data[dataset_name] = {}
        param_options = dataset_param_options[dataset_name]
        objectives = dataset_to_obj[dataset_name]
        
        if isinstance(objectives, dict):
            objectives = objectives["objectives"]
        else:
            objectives = [objectives]
        
        for method_path, _ in optimization_data.items():
            method_name = method_path.split('/')[-1]
            parameter_entropy_data[dataset_name][method_name] = {}
            
            # Initialize parameter entropy storage
            for param_name in param_options.keys():
                parameter_entropy_data[dataset_name][method_name][param_name] = []
            
            # Get all run directories for this method
            runs = glob.glob(method_path + "/run_*/")
            
            for run_dir in runs:
                try:
                    # Load seen observations for this run
                    with open(os.path.join(run_dir, "seen_observations.json"), "r") as f:
                        seen_observations = json.load(f)
                    
                    df = pd.DataFrame(seen_observations)
                    
                    # Group observations similar to main approach
                    if "llm" in method_path:
                        group_cols = ["reasoning", "explanation"]
                    elif "bayesian" in method_path:
                        all_cols = df.columns.tolist()
                        group_cols = [c for c in all_cols if c not in objectives]
                    
                    # Extract parameter selections for this run
                    grouped_list = list(df.groupby(group_cols, sort=False))
                    run_selections = []
                    
                    for group in grouped_list:
                        group_df = group[1]
                        selection_dict = {}
                        for param_name in param_options.keys():
                            if param_name in group_df.columns:
                                all_values = group_df[param_name].tolist()
                                selection_dict[param_name] = all_values[0]
                        run_selections.append(selection_dict)
                    
                    # Skip first selection for Bayesian methods
                    if "bayesian" in method_path:
                        run_selections = run_selections[1:]
                    
                    # Compute parameter-specific entropies for this run
                    param_entropies = compute_individual_run_parameter_entropies(run_selections, param_options)
                    
                    # Store entropy for each parameter
                    for param_name, entropy_val in param_entropies.items():
                        parameter_entropy_data[dataset_name][method_name][param_name].append(entropy_val)
                    
                except Exception as e:
                    print(f'Error processing {run_dir}: {e}')
                    continue
            

    
    return parameter_entropy_data

def create_cumulative_entropy_si_figures(parameter_entropy_data, provider_models):
    """
    Create SI figures showing cumulative entropy for each parameter separately.
    Similar to original create_entropy_figures but for cumulative approach.
    """
    import math
    
    ALL_METHODS = [item for sublist in provider_models.values() for item in sublist]
    dataset_figures = {}
    
    for dataset_name, dataset_data in parameter_entropy_data.items():
        # Clean method names and filter valid methods
        valid_methods = {}
        for method_path, param_data in dataset_data.items():
            # Clean method name similar to original approach
            clean_method = method_path.replace('-1-20-20', '').replace('-latest', '')
            clean_method = clean_method.replace('-preview-03-25', '').replace('-preview-04-17', '')
            clean_method = clean_method.replace('-20250514', '').replace('-des0', '-des')
            clean_method = clean_method.replace('-preview-06-17', '')
            clean_method = clean_method.replace('-20250805', '')
            clean_method = clean_method.replace('-20250514', '').replace('-des0', '-des')
            clean_method = clean_method.replace('-preview-06-17', '')
            
            if clean_method in ALL_METHODS:
                # Only include methods that have data for at least one parameter
                if any(len(param_entropies) > 0 for param_entropies in param_data.values()):
                    valid_methods[clean_method] = param_data
        
        if not valid_methods:
            continue
        
        # Sort methods according to ALL_METHODS order
        ordered_methods = [(method, valid_methods[method]) for method in ALL_METHODS if method in valid_methods]
        
        # Calculate grid dimensions
        n_methods = len(ordered_methods)
        n_cols = min(6, n_methods)  # Maximum 6 columns
        n_rows = math.ceil(n_methods / n_cols)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'\n{dataset_name} - Cumulative Parameter Entropy by Method\n', fontsize=24)
        
        # Handle single subplot case
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_methods > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Get parameter names for consistent coloring across all subplots
        all_param_names = set()
        for method, param_data in ordered_methods:
            all_param_names.update(param_data.keys())
        all_param_names = sorted(list(all_param_names))
        param_colors = {param: plt.cm.Set2(i/len(all_param_names)) for i, param in enumerate(all_param_names)}
        
        # Plot each method
        for idx, (method, param_data) in enumerate(ordered_methods):
            ax = axes[idx]
            
            # Create boxplots for each parameter's cumulative entropy distribution
            param_positions = []
            param_labels = []
            param_box_data = []
            
            for param_idx, (param_name, entropy_values) in enumerate(param_data.items()):
                if len(entropy_values) > 0:  # Only plot if there's data
                    param_positions.append(param_idx + 1)
                    param_labels.append(param_name)
                    param_box_data.append(entropy_values)
            
            if param_box_data:
                # Create boxplots
                bp = ax.boxplot(param_box_data, positions=param_positions, patch_artist=True, 
                               widths=0.6, showfliers=True)
                
                # Color the boxplots
                for patch, param_name in zip(bp['boxes'], param_labels):
                    patch.set_facecolor(param_colors[param_name])
                    patch.set_alpha(0.7)
                
                # Customize boxplot appearance
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black')
                
                ax.set_xticks(param_positions)
                ax.set_xticklabels(param_labels, rotation=45, ha='right')
            
            ax.set_title(f'{method}', fontweight='bold')
            ax.set_ylabel('Cumulative Entropy')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)  # Make room for title and labels
        dataset_figures[dataset_name] = fig
    
    return dataset_figures

def analyze_entropy_median_distribution(entropy_data):
    """
    Analyze the distribution of median cumulative entropy values across methods
    to identify which methods might stand out as significantly different.
    """
    print("\n" + "="*60)
    print("CUMULATIVE ENTROPY MEDIAN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Collect all median values across datasets and methods
    all_medians = []
    method_medians = defaultdict(list)  # method -> list of medians across datasets
    dataset_medians = defaultdict(list)  # dataset -> list of medians across methods
    
    for dataset_name, methods_data in entropy_data.items():
        dataset_name_display = dataset_name.replace('_', ' ').title()
        print(f"\n📊 {dataset_name_display}:")
        
        dataset_values = []
        for method_path, data in methods_data.items():
            if data['median'] is not None:
                median_val = data['median']
                method_name = method_path.split('/')[-1].replace('-1-20-20', '').replace('-des0', '-des')
                
                # Get simplified method name (commented out due to missing method_name_mapping)
                # for full_name, short_name in method_name_mapping.items():
                #     if full_name in method_name:
                #         method_name = short_name
                #         break
                
                all_medians.append(median_val)
                method_medians[method_name].append(median_val)
                dataset_values.append((method_name, median_val))
        
        # Sort and show top/bottom methods for this dataset
        dataset_values.sort(key=lambda x: x[1], reverse=True)
        dataset_medians[dataset_name] = [val for _, val in dataset_values]
        
        print(f"   🔝 Highest Entropy: {dataset_values[0][0]} ({dataset_values[0][1]:.3f})")
        print(f"   🔻 Lowest Entropy:  {dataset_values[-1][0]} ({dataset_values[-1][1]:.3f})")
        print(f"   📈 Range: {dataset_values[0][1] - dataset_values[-1][1]:.3f}")
        print(f"   📊 Dataset Median: {np.median([val for _, val in dataset_values]):.3f}")
    
    # Overall statistics
    print(f"\n🌍 OVERALL STATISTICS:")
    print(f"   Total median values: {len(all_medians)}")
    print(f"   Global median: {np.median(all_medians):.3f}")
    print(f"   Global mean: {np.mean(all_medians):.3f}")
    print(f"   Global std: {np.std(all_medians):.3f}")
    print(f"   Global range: {np.max(all_medians) - np.min(all_medians):.3f}")
    
    # Method-level analysis (average across datasets)
    print(f"\n🔬 METHOD-LEVEL ANALYSIS (Average Across Datasets):")
    method_avg_medians = []
    for method_name, medians in method_medians.items():
        if len(medians) > 0:
            avg_median = np.mean(medians)
            std_median = np.std(medians) if len(medians) > 1 else 0
            method_avg_medians.append((method_name, avg_median, std_median, len(medians)))
    
    # Sort by average median
    method_avg_medians.sort(key=lambda x: x[1], reverse=True)
    
    print("   Rank | Method | Avg Median | Std | Datasets")
    print("   -----|---------|-----------|-----|----------")
    for i, (method, avg_med, std_med, n_datasets) in enumerate(method_avg_medians):
        print(f"   {i+1:4d} | {method:15s} | {avg_med:8.3f} | {std_med:.3f} | {n_datasets}")
    
    # Identify outliers using z-score
    global_mean = np.mean(all_medians)
    global_std = np.std(all_medians)
    threshold = 2.0  # 2 standard deviations
    
    print(f"\n🚨 OUTLIER METHODS (|z-score| > {threshold}):")
    outlier_found = False
    for method, avg_med, std_med, n_datasets in method_avg_medians:
        z_score = (avg_med - global_mean) / global_std if global_std > 0 else 0
        if abs(z_score) > threshold:
            outlier_found = True
            direction = "HIGH" if z_score > 0 else "LOW"
            print(f"   🎯 {method}: {avg_med:.3f} (z={z_score:+.2f}, {direction} entropy)")
    
    if not outlier_found:
        print("   No significant outliers found.")
    
    # Dataset-level variance analysis
    print(f"\n📊 DATASET-LEVEL VARIANCE:")
    for dataset_name, values in dataset_medians.items():
        dataset_std = np.std(values)
        dataset_range = np.max(values) - np.min(values)
        print(f"   {dataset_name.replace('_', ' ').title():20s}: std={dataset_std:.3f}, range={dataset_range:.3f}")
    
    return {
        'all_medians': all_medians,
        'method_medians': method_medians,
        'dataset_medians': dataset_medians,
        'method_rankings': method_avg_medians,
        'global_stats': {
            'median': np.median(all_medians),
            'mean': np.mean(all_medians),
            'std': np.std(all_medians)
        }
    }

if __name__ == "__main__":
    # Get run path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')
    
    n_tracks = 20
    track_size = 20

    # Load path_dict similar to figure_5.py and figure_7.py
    path_dict = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        bo_path = os.path.join(run_path, f'bayesian/{dataset_name}/benchmark/')
        llm_path = os.path.join(run_path, f'llm/{dataset_name}/benchmark/')
        
        if os.path.exists(bo_path):
            bo_paths = [os.path.join(bo_path, path) for path in os.listdir(bo_path)]
            for path in bo_paths:
                if path.endswith('-20') or path.endswith('-20-des0'):
                    if os.path.exists(path):
                        path_dict[dataset_name][path] = None  # We don't need tracks, just path info
        
        if os.path.exists(llm_path):
            llm_paths = [os.path.join(llm_path, path) for path in os.listdir(llm_path)]
            for path in llm_paths:
                if 'claude' in path and 'gpt' in path:
                    continue
                if '1-20-20' in path:
                    if os.path.exists(path):
                        path_dict[dataset_name][path] = None  # We don't need tracks, just path info

    # Get parameter options for each dataset
    dataset_param_options = {}
    for dataset_name in dataset_names:
        param_options = {}
        dataset = Dataset(dataset_name)
        for param in dataset.param_space:
            param_options[param.name] = param.options
        dataset_param_options[dataset_name] = param_options

    print("Extracting cumulative entropy data...")
    entropy_data = extract_cumulative_entropy_data(path_dict, dataset_param_options)

    # Sort by dataset_name based on how it appears in dataset_to_color
    sorted_dataset_names = sorted(dataset_to_color.keys(), key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), reverse=True)
    entropy_data = {k: entropy_data[k] for k in sorted_dataset_names if k in entropy_data}
    
    # Group methods by provider for boxplots
    method_names = []
    remove_datasets = []

    # First collect all method names
    for dataset_name, dataset_data in entropy_data.items():
        for key in dataset_data.keys():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '')
            method_name = method_name.replace('-latest', '')
            method_name = method_name.replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '')
            method_name = method_name.replace('-preview-04-17', '')
            method_name = method_name.replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
            if 'gpt-4.1' in method_name and ('nano' in method_name):
                continue
            method_names.append(method_name)

    # Remove duplicates while preserving order
    method_names = list(dict.fromkeys(method_names))

    # Group methods by provider
    provider_methods = {}
    for provider, models_list in model_to_provider.items():
        # Only include methods that are in our filtered method_names list
        provider_filtered_methods = [method for method in models_list if method in method_names]
        if provider_filtered_methods:  # Only include providers that have methods
            provider_methods[provider] = provider_filtered_methods
        else:
            print(f'No methods found for provider: {provider}')

    # Create individual plots for each provider
    all_provider_data = {}
    for provider_name, methods in provider_methods.items():
        print(f"\nCreating cumulative entropy plot for {provider_name}...")
        fig, ax, provider_data = create_individual_provider_entropy_plot(provider_name, methods, entropy_data, remove_datasets)
        
        if fig is not None:
            all_provider_data[provider_name] = provider_data
            # Save the figure to ./pngs/figure_7_new_{provider_name}.png
            os.makedirs('./pngs', exist_ok=True)
            plt.savefig(f'./pngs/figure_7_{provider_name}.png', dpi=300, bbox_inches='tight')
        
    print(f"\nCreated {len(provider_methods)} individual provider cumulative entropy plots.")

    # Print summary
    print(f"\nSummary of plotted providers:")
    for provider, methods in provider_methods.items():
        if provider in all_provider_data:
            print(f"  {provider}: {len(methods)} methods - {', '.join(methods)}")

    # Generate SI figures showing parameter-specific cumulative entropy
    print("\nGenerating SI figures with parameter-specific cumulative entropy...")
    parameter_entropy_data = extract_parameter_specific_entropy_data(path_dict, dataset_param_options)
    
    # Create SI figures
    provider_models = model_to_provider
    
    dataset_si_figures = create_cumulative_entropy_si_figures(parameter_entropy_data, provider_models)
    
    for dataset_name, fig in dataset_si_figures.items():
        # Save the figure to ./pngs/figure_SI_cumulative_entropy_{dataset_name}.png
        os.makedirs('./pngs', exist_ok=True)
        fig.savefig(f'./pngs/figure_S10_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
    print(f"\nCreated {len(dataset_si_figures)} SI figures with parameter-specific cumulative entropy.")
    
    print(f'Figure 7 Anthropic saved to ./pngs/figure_7_Anthropic.png')
    print(f'Figure 7 Google saved to ./pngs/figure_7_Google.png')
    print(f'Figure 7 OpenAI saved to ./pngs/figure_7_OpenAI.png')
    print(f'Figure 7 Atlas saved to ./pngs/figure_7_Atlas.png')

    print(f'Figure S10 Buchwald_Hartwig saved to ./pngs/figure_S10_Buchwald_Hartwig.png')
    print(f'Figure S10 Suzuki_Doyle saved to ./pngs/figure_S10_Suzuki_Doyle.png')
    print(f'Figure S10 Suzuki_Cernak saved to ./pngs/figure_S10_Suzuki_Cernak.png')
    print(f'Figure S10 Reductive_Amination saved to ./pngs/figure_S10_Reductive_Amination.png')
    print(f'Figure S10 amide_coupling_hte saved to ./pngs/figure_S10_amide_coupling_hte.png')
    print(f'Figure S10 Chan_Lam_Full saved to ./pngs/figure_S10_Chan_Lam_Full.png')