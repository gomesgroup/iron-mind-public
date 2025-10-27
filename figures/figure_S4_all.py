import pandas as pd
import glob, json, os
import numpy as np
import sys

import matplotlib.pyplot as plt

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
    ],
    "Random": [
        'random',
    ]
}


dataset_to_color = {
    'suzuki_cernak': '#f0e142',      # Lighter gray
    'amide_coupling_hte': '#d55e00', # Yellow  
    'reductive_amination': '#cc797f', # Orange
    'suzuki_doyle': '#009e74',      # Light Blue
    'chan_lam_full': '#0071b2',     # Dark Orange
    'buchwald_hartwig': '#000000',  # Dark blue (darkest)
}
dataset_order = list(dataset_to_color.keys())

def get_objective_value(obs_or_group, dataset_config, aggregation='first'):
    """
    Extract objective value from observation(s) based on dataset configuration
    
    Args:
        obs_or_group: Single observation dict or pandas group for aggregation
        dataset_config: Dataset configuration (string or dict)
        aggregation: How to aggregate multiple observations ('first', 'mean', 'max', 'min', 'median')
    """
    if isinstance(dataset_config, str):
        # Simple single objective
        if hasattr(obs_or_group, 'iloc'):  # It's a pandas group
            if aggregation == 'first':
                return obs_or_group[dataset_config].iloc[0]
            elif aggregation == 'mean':
                return obs_or_group[dataset_config].mean()
            elif aggregation == 'max':
                return obs_or_group[dataset_config].max()
            elif aggregation == 'min':
                return obs_or_group[dataset_config].min()
            elif aggregation == 'median':
                return obs_or_group[dataset_config].median()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Single observation
            return obs_or_group[dataset_config]
    else:
        # Complex multi-objective
        if hasattr(obs_or_group, 'iloc'):  # It's a pandas group
            if aggregation == 'first':
                objectives = [obs_or_group[obj].iloc[0] for obj in dataset_config['objectives']]
            elif aggregation == 'mean':
                objectives = [obs_or_group[obj].mean() for obj in dataset_config['objectives']]
            elif aggregation == 'max':
                objectives = [obs_or_group[obj].max() for obj in dataset_config['objectives']]
            elif aggregation == 'min':
                objectives = [obs_or_group[obj].min() for obj in dataset_config['objectives']]
            elif aggregation == 'median':
                objectives = [obs_or_group[obj].median() for obj in dataset_config['objectives']]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Single observation
            objectives = [obs_or_group[obj] for obj in dataset_config['objectives']]
        
        # Apply transformation
        if dataset_config['transform'] == 'subtract':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] - objectives[order[1]]
        elif dataset_config['transform'] == 'add':
            return sum(objectives)
        elif dataset_config['transform'] == 'multiply':
            result = 1
            for obj in objectives:
                result *= obj
            return result
        elif dataset_config['transform'] == 'divide':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] / objectives[order[1]] if objectives[order[1]] != 0 else 0
        elif dataset_config['transform'] == 'ratio':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] / (objectives[order[0]] + objectives[order[1]]) if (objectives[order[0]] + objectives[order[1]]) != 0 else 0
        elif dataset_config['transform'] == 'weighted_selectivity':
            order = dataset_config.get('order', [0, 1])
            desired = objectives[order[0]]
            undesired = objectives[order[1]]
            total = desired + undesired
            if total > 0:
                selectivity_ratio = desired / total
                return selectivity_ratio * desired
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown transform: {dataset_config['transform']}")

def get_tracks(path, dataset_name, bo=False, n_tracks=None, track_size=None, aggregation='first', return_rundir=True):
    """
    Extract optimization tracks from results directory
    
    Args:
        path: Directory containing run_* subdirectories
        dataset_name: Name of dataset to get objective configuration
        bo: If True, skip first observation (for BO baselines)
        n_tracks: Maximum number of tracks to return
        track_size: Maximum length of each track
        aggregation: How to aggregate multiple measurements per parameter config ('first', 'mean', 'max', 'min', 'median')
    
    Returns:
        List of arrays, each showing cumulative best objective value over iterations
    """
    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if 'run_' in d]
    # Sort run_dirs by run_*
    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]))
    tracks = []
    
    # Get dataset configuration
    dataset_config = dataset_to_obj[dataset_name]
    
    for rd in run_dirs:
        # Load 'seen_observations.json'
        seen_path = os.path.join(rd, 'seen_observations.json')
        objective_values = []
        
        if os.path.exists(seen_path):
            with open(seen_path, 'r') as f:
                seen_data = json.load(f)
            
            if dataset_name == 'Chan_Lam_Full':
                # Special handling for Chan_Lam_Full - group by reasoning for LLM, params for BO
                import pandas as pd
                df = pd.DataFrame(seen_data)
                
                # Check if this is LLM (has reasoning) or BO method
                if 'reasoning' in df.columns and df['reasoning'].notna().any():
                    # LLM method - group by reasoning
                    param_groups = df.groupby('reasoning')
                else:
                    # BO method - group by parameter names
                    exclude_keys = {'desired_yield', 'undesired_yield'}
                    param_names = [col for col in df.columns if col not in exclude_keys]
                    param_groups = df.groupby(param_names)
                
                # Process groups and extract objectives with min aggregation
                for name, group in param_groups:
                    try:
                        # Calculate weighted selectivity for each measurement in the group
                        desired_yields = group['desired_yield'].values
                        undesired_yields = group['undesired_yield'].values
                        
                        # Calculate weighted selectivity: (desired/(desired + undesired)) * desired
                        weighted_selectivities = []
                        for desired, undesired in zip(desired_yields, undesired_yields):
                            total = desired + undesired
                            if total > 0:
                                selectivity_ratio = desired / total
                                weighted_selectivity = selectivity_ratio * desired
                                weighted_selectivities.append(weighted_selectivity)
                            else:
                                weighted_selectivities.append(0.0)
                        
                        # Use minimum weighted selectivity for the parameter group (worst-case/lower bound)
                        if weighted_selectivities:
                            obj_value = np.min(weighted_selectivities)
                        else:
                            obj_value = 0.0
                            
                        if np.isnan(obj_value):
                            objective_values.append(0)
                        else:
                            objective_values.append(obj_value)
                    except (KeyError, TypeError, ZeroDivisionError):
                        objective_values.append(0)
            else:
                # Standard handling for other datasets
                for obs in seen_data:
                    try:
                        obj_value = get_objective_value(obs, dataset_config, aggregation)
                        if np.isnan(obj_value):
                            objective_values.append(0)
                        else:
                            objective_values.append(obj_value)
                    except (KeyError, TypeError, ZeroDivisionError):
                        # Handle missing keys or invalid operations
                        objective_values.append(0)
        
        # Convert to array and handle BO case
        objective_array = np.array(objective_values)
        if bo:
            # Skip the first observation
            objective_array = objective_array[1:].round(2)
        
        # Truncate if track_size specified
        if track_size is not None:
            objective_array = objective_array[:track_size]
        
        # Create cumulative maximum track
        if len(objective_array) > 0:
            tracks.append(np.maximum.accumulate(objective_array))
        else:
            tracks.append(np.array([]))
    
    # Limit number of tracks if specified
    if n_tracks is not None:
        if len(tracks) < n_tracks:
            return None
        else:
            tracks = tracks[:n_tracks]
    
    if return_rundir:
        return tracks, run_dirs
    else:
        return tracks

def get_convergence_data(path_dict):
    convergence_data = {}
    for dataset_name, optimization_data in path_dict.items():
        convergence_data[dataset_name.lower()] = {}
        for key, tracks in optimization_data.items():
            convergence_data[dataset_name.lower()][key] = {'convergence_indices': [], 'median': None, 'q1': None, 'q3': None}
            # Get the index of the max in each row
            try:
                top_obs_idx = [np.argmax(track) for track in tracks]
                convergence_data[dataset_name.lower()][key]['convergence_indices'] = top_obs_idx
                
                median = float(np.median(top_obs_idx))
                q1 = float(np.percentile(top_obs_idx, 25))
                q3 = float(np.percentile(top_obs_idx, 75))
                convergence_data[dataset_name.lower()][key]['median'] = median
                convergence_data[dataset_name.lower()][key]['q1'] = q1
                convergence_data[dataset_name.lower()][key]['q3'] = q3
            except Exception as e:
                print(f'Error getting convergence data for {key} in {dataset_name}: {e}')
    return convergence_data

# Function to plot boxplots for a provider
def plot_provider_boxplots(ax, provider_name, provider_method_list, remove_datasets=[]):
    if not provider_method_list:
        ax.axis('off')
        return
    
    convergence_data_refined = {k: v for k, v in convergence_data.items() if k not in remove_datasets}
    # Add horizontal line at y=0 in the background
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    
    # Create mapping of method names to their positions
    method_positions = {name: idx * 2 + 1 for idx, name in enumerate(provider_method_list)}
    
    for method_name in provider_method_list:
        position = method_positions[method_name]
        offset = 0
        for dataset_name, dataset_data in convergence_data_refined.items():
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
                
                if current_method == method_name:
                    boxplot = ax.boxplot(
                        method_data['convergence_indices'],
                        positions=[position + offset],
                        widths=0.2,
                        patch_artist=True,
                        showfliers=True,
                        zorder=1
                    )

                    # Print the IQR for the method and dataset
                    q1 = np.percentile(method_data['convergence_indices'], 25)
                    q3 = np.percentile(method_data['convergence_indices'], 75)
                    IQR = q3 - q1
                    # print(f"{method_name} - {dataset_name}: {IQR:.2f}")
                    
                    # Customize boxplot appearance
                    for box in boxplot['boxes']:
                        box.set(color='black', linewidth=2)
                        box.set(facecolor=dataset_to_color[dataset_name.lower()], alpha=0.8)
                    
                    for whisker in boxplot['whiskers']:
                        whisker.set(color='black', linewidth=2)
                        
                    for cap in boxplot['caps']:
                        cap.set(color='black', linewidth=2)
                        
                    for median in boxplot['medians']:
                        if IQR == 0:
                            median.set(color='k', linewidth=4)
                        else:
                            median.set(color='limegreen', linewidth=4)
                        
                    for flier in boxplot['fliers']:
                        flier.set(marker='o', markerfacecolor=dataset_to_color[dataset_name.lower()],
                                markeredgecolor='black', markersize=12)
                    
                    offset += 0.3

    # Calculate centered tick positions
    centered_positions = []
    for method_name in provider_method_list:
        base_position = method_positions[method_name]
        # Count how many datasets have data for this method
        n_datasets = 0
        for dataset_name, dataset_data in convergence_data_refined.items():
            for key in dataset_data.keys():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '')
                current_method = current_method.replace('-latest', '')
                current_method = current_method.replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '')
                current_method = current_method.replace('-preview-04-17', '')
                current_method = current_method.replace('-des0', '-des')
                current_method = current_method.replace('-20250805', '')
                if current_method == method_name:
                    n_datasets += 1
                    break  # Only count once per dataset
        
        # Calculate center position: base + (n_datasets - 1) * 0.35 / 2
        center_offset = (n_datasets - 1) * 0.3 / 2
        centered_positions.append(base_position + center_offset)
    
    # ADD VERTICAL SEPARATOR LINES HERE
    # Add vertical dashed lines between method groups
    if len(provider_method_list) > 1:
        for i in range(len(provider_method_list) - 1):
            # Calculate midpoint between current and next method
            separator_x = (centered_positions[i] + centered_positions[i + 1]) / 2
            ax.axvline(x=separator_x, color='k', linestyle='-', linewidth=4, alpha=1.0, zorder=0)
            
    # Configure the plot
    ax.set_xticks(centered_positions)
    # For any method name with -thinking or -medium, do /n
    xticklabels = []
    for method in provider_method_list:
        if '-thinking' in method or '-medium' in method:
            xticklabels.append("-".join(method.split('-')[:-1]) + "\n" + method.split('-')[-1])
        else:
            xticklabels.append(method)
    ax.set_xticklabels(xticklabels, rotation=0, ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(False, axis='x')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    
    # Add title for the provider
    # ax.set_title(provider_name, fontsize=20, fontweight='bold', pad=20)

    # Make plot border black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # Set top spine to be invisible
    ax.spines['top'].set_visible(False)

def create_individual_provider_plot(provider_name, provider_method_list, remove_datasets=[]):
    """Create a single plot for one provider"""
    if not provider_method_list:
        print(f"No methods found for {provider_name}")
        return
    
    convergence_data_refined = {k: v for k, v in convergence_data.items() if k not in remove_datasets}
    
    # Fixed figure sizing for consistent 2x2 layout
    fig_width = 15
    fig_height = 8
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Add horizontal line at y=0 in the background
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    
    # Create mapping of method names to their positions with better spacing
    method_positions = {name: idx * 3 + 1 for idx, name in enumerate(provider_method_list)}
    
    provider_data = {}
    for method_name in provider_method_list:
        # print(method_name)
        position = method_positions[method_name]
        offset = 0
        
        for dataset_name, dataset_data in convergence_data_refined.items():
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
                
                if current_method == method_name:
                    provider_data[dataset_name][method_name] = method_data['convergence_indices']
                    boxplot = ax.boxplot(
                        method_data['convergence_indices'],
                        positions=[position + offset],
                        widths=0.3,
                        patch_artist=True,
                        showfliers=True,
                        zorder=3
                    )
                    IQR = np.percentile(method_data['convergence_indices'], 75) - np.percentile(method_data['convergence_indices'], 25)
                    
                    # Customize boxplot appearance
                    # Set median properties first so it appears behind the box
                    for median in boxplot['medians']:
                        if IQR == 0 and np.median(method_data['convergence_indices']) == 100:
                            median.set(color='k', linewidth=4, zorder=2)
                            # Place a black box behind the asterisk
                            from matplotlib.patches import Rectangle
                            median_x = median.get_xdata()[0]
                            median_y = median.get_ydata()[0]
                            box_width = 0.3
                            box_height = 3.5
                            rect = Rectangle((median_x, median_y + box_height/2), box_width, box_height, facecolor='black', edgecolor='white', alpha=1.0, zorder=3, linewidth=2)
                            ax.add_patch(rect)
                            # Place an asterisk right above the median line
                            ax.text(median_x + box_width/2, median_y + box_height/2, '*', fontsize=30, ha='center', va='center', zorder=4, color='white')
                        elif IQR == 0:
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
    
    # Calculate centered tick positions
    centered_positions = []
    for method_name in provider_method_list:
        base_position = method_positions[method_name]
        # Count how many datasets have data for this method
        n_datasets = 0
        for dataset_name, dataset_data in convergence_data_refined.items():
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
    ax.set_ylim(0, 21)
    # increment by 2
    ax.set_yticks(np.arange(0, 21, 2))
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    
    # Make plot border black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, provider_data

if __name__ == "__main__":
    # user input for the run path
    # Get run path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')
    bo_benchmark_path = os.path.join(run_path, 'bayesian/{dataset_name}/benchmark/')
    llm_benchmark_path = os.path.join(run_path, 'llm/{dataset_name}/benchmark/')
    n_tracks = 20
    track_size = 20

    path_dict = {}
    path_dict_with_run_paths = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        path_dict_with_run_paths[dataset_name] = {}
        bo_path = bo_benchmark_path.format(dataset_name=dataset_name)
        llm_path = llm_benchmark_path.format(dataset_name=dataset_name)
        
        bo_paths = [os.path.join(bo_path, path) for path in os.listdir(bo_path)]
        llm_paths = [os.path.join(llm_path, path) for path in os.listdir(llm_path)]

        for path in bo_paths:
            if path.endswith('-20') or path.endswith('-20-des0'):
                track_data, run_dirs = get_tracks(path, dataset_name, bo=True, n_tracks=n_tracks, track_size=track_size, return_rundir=True)
                if track_data is not None:
                    path_dict[dataset_name][path] = track_data
                    path_dict_with_run_paths[dataset_name][path] = {}
                    for track, run_dir in zip(track_data, run_dirs):
                        path_dict_with_run_paths[dataset_name][path][run_dir] = track
        
        for path in llm_paths:
            if 'claude' in path and 'gpt' in path:
                continue
            if '1-20-20' in path:
                try:
                    track_data, run_dirs = get_tracks(path, dataset_name, n_tracks=n_tracks, track_size=track_size, return_rundir=True)
                except Exception as e:
                    print(f'{path} - {e}')
                    continue
                if track_data is not None:
                    path_dict[dataset_name][path] = track_data
                    path_dict_with_run_paths[dataset_name][path] = {}
                    for track, run_dir in zip(track_data, run_dirs):
                        path_dict_with_run_paths[dataset_name][path][run_dir] = track

    for dataset_name, paths in path_dict.items():
        print(f"{dataset_name}: {len(paths)}")

    convergence_data = get_convergence_data(path_dict)

    # Sort by dataset_name based on how it appears in dataset_to_color
    sorted_dataset_names = sorted(dataset_to_color.keys(), key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), reverse=True)

    # Sort top_5_dict by the sorted dataset_names
    convergence_data = {k: convergence_data[k] for k in sorted_dataset_names}
    
    # Group methods by provider for boxplots
    method_data = {}
    method_names = []
    # FILTER_METHODS = False  # Set to False to include all methods
    remove_datasets = []

    # First collect all method names and their dataset coverage
    for dataset_name, dataset_data in convergence_data.items():
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
    remove_datasets = []

    all_provider_data = {}
    for provider_name, methods in provider_methods.items():
        print(f"\nCreating plot for {provider_name}...")
        fig, ax, provider_data = create_individual_provider_plot(provider_name, methods, remove_datasets)
        
        all_provider_data[provider_name] = provider_data
        # Save the figure to ./pngs/figure_5_{provider_name}.png
        os.makedirs('./pngs', exist_ok=True)
        plt.savefig(f'./pngs/figure_S4_{provider_name}.png', dpi=300, bbox_inches='tight')
        
    print(f"\nCreated {len(provider_methods)} individual provider plots.")

    # Print summary
    print(f"\nSummary of plotted providers:")
    for provider, methods in provider_methods.items():
        print(f"  {provider}: {len(methods)} methods - {', '.join(methods)}")

    print(f'Figure S4 Anthropic saved to ./pngs/figure_S4_Anthropic.png')
    print(f'Figure S4 Google saved to ./pngs/figure_S4_Google.png')
    print(f'Figure S4 OpenAI saved to ./pngs/figure_S4_OpenAI.png')
    print(f'Figure S4 Atlas saved to ./pngs/figure_S4_Atlas.png')    