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
    'buchwald_hartwig': '#000000',  # Dark blue (darkest)
    'chan_lam_full': '#0071b2',     # Dark Orange
    'suzuki_doyle': '#009e74',      # Light Blue
    'reductive_amination': '#cc797f', # Orange
    'amide_coupling_hte': '#d55e00', # Yellow  
    'suzuki_cernak': '#f0e142',      # Lighter gray
}
dataset_order = list(dataset_to_color.keys())

# Dataset-specific maximum values for threshold calculations
dataset_to_max = {
    'buchwald_hartwig': 100.0,
    'suzuki_doyle': 100.0,
    'suzuki_cernak': 100.0,
    'reductive_amination': 100.0,
    'amide_coupling_hte': 100.0,
    'chan_lam_full': 86.39,  # best weighted selectivity with min aggregation
}

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

def calculate_time_to_threshold(track, threshold_percent, dataset_max=100.0):
    """
    Calculate the iteration at which the track reaches a percentage of the dataset maximum (100%)
    
    Args:
        track: Array of cumulative best values
        threshold_percent: Percentage of dataset maximum (0.8 for 80%, 0.95 for 95%)
        dataset_max: Maximum possible value for the dataset (default 100.0)
    
    Returns:
        Iteration number (1-based) when threshold is reached, or None if never reached
    """
    if len(track) == 0:
        return None
    
    # Use dataset maximum (100%), not the run's maximum
    threshold_value = dataset_max * threshold_percent
    
    # Find first iteration where threshold is reached
    threshold_indices = np.where(track >= threshold_value)[0]
    
    if len(threshold_indices) > 0:
        return threshold_indices[0] + 1  # Convert to 1-based indexing
    else:
        return None  # Never reached threshold

def get_time_to_best_data(path_dict, threshold_percent=0.8):
    """Extract time-to-threshold data for all methods and datasets"""
    time_to_best_data = {}
    
    for dataset_name, optimization_data in path_dict.items():
        time_to_best_data[dataset_name.lower()] = {}
        
        # Get the correct dataset maximum
        dataset_max = dataset_to_max.get(dataset_name.lower(), 100.0)
        
        for method_key, tracks in optimization_data.items():
            times = []
            
            for track in tracks:
                time_to_threshold = calculate_time_to_threshold(track, threshold_percent, dataset_max=dataset_max)
                if time_to_threshold is not None:
                    times.append(time_to_threshold)
            
            if times:  # Only store if we have valid times (exclude runs that never reach threshold)
                time_to_best_data[dataset_name.lower()][method_key] = {
                    'times': times,
                    'n_runs': len(times),  # Number of runs that reached threshold
                    'total_runs': len(tracks),  # Total number of runs
                    'median': np.median(times),
                    'q1': np.percentile(times, 25),
                    'q3': np.percentile(times, 75)
                }
    
    return time_to_best_data

def create_time_to_best_plot(time_data, threshold_percent, save_path="./pngs/"):
    """Create boxplots for time-to-threshold analysis grouped by provider"""
    
    # Sort datasets by color order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Filter data to only include datasets we have
    filtered_time_data = {k: v for k, v in time_data.items() if k in sorted_dataset_names}
    
    # Group methods by provider
    provider_methods = {}
    all_methods = set()
    
    for dataset_data in filtered_time_data.values():
        for method_key in dataset_data.keys():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
                
            all_methods.add(method_name)
    
    # Group by provider
    for provider, methods_list in model_to_provider.items():
        provider_filtered_methods = [method for method in methods_list if method in all_methods]
        if provider_filtered_methods:
            provider_methods[provider] = provider_filtered_methods
    
    # Create individual plots for each provider
    for provider_name, methods in provider_methods.items():
        if not methods:
            continue
            
        print(f"\nCreating time-to-{int(threshold_percent*100)}% plot for {provider_name}...")
        
        # Create figure
        fig_width = 15
        fig_height = 8
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        
        # Create mapping of method names to their positions
        method_positions = {name: idx * 3 + 1 for idx, name in enumerate(methods)}
        
        for method_name in methods:
            position = method_positions[method_name]
            offset = 0
            
            for dataset_key in sorted_dataset_names:
                if dataset_key not in filtered_time_data:
                    continue
                    
                dataset_data = filtered_time_data[dataset_key]
                
                # Find data for this method in this dataset
                method_data = None
                for method_key, data in dataset_data.items():
                    key_method = method_key.split('/')[-1]
                    key_method = key_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                    key_method = key_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                    key_method = key_method.replace('-preview-06-17', '')
                    key_method = key_method.replace('-20250805', '')
                    if key_method == method_name:
                        method_data = data
                        break
                
                if method_data is not None and len(method_data) > 0:
                    # Check if we have only one data point
                    if len(method_data['times']) == 1:
                        # Plot single point as a circle (like an outlier)
                        ax.scatter(
                            [position + offset], 
                            method_data['times'], 
                            marker='o', 
                            facecolor=dataset_to_color[dataset_key],
                            edgecolor='black', 
                            s=64,  # markersize=8 equivalent
                            alpha=0.8,
                            zorder=3
                        )
                    else:
                        # Use normal boxplot for multiple points
                        boxplot = ax.boxplot(
                            method_data['times'],
                            positions=[position + offset],
                            widths=0.3,
                            patch_artist=True,
                            showfliers=True,
                            zorder=3
                        )
                        
                        # Customize boxplot appearance
                        for box in boxplot['boxes']:
                            box.set(color='black', linewidth=2, zorder=3)
                            box.set(facecolor=dataset_to_color[dataset_key], alpha=0.8, zorder=1)
                        
                        for whisker in boxplot['whiskers']:
                            whisker.set(color='black', linewidth=2, zorder=1)
                        
                        for cap in boxplot['caps']:
                            cap.set(color='black', linewidth=2, zorder=1)
                        
                        for median in boxplot['medians']:
                            # Check if all values are the same (flat distribution)
                            if len(set(method_data['times'])) == 1:
                                # All iterations are the same - use black median line
                                median.set(color='black', linewidth=4, zorder=2)
                            else:
                                # Normal case - use white median line
                                median.set(color='white', linewidth=4, zorder=2)
                        
                        for flier in boxplot['fliers']:
                            flier.set(marker='o', markerfacecolor=dataset_to_color[dataset_key],
                                    markeredgecolor='black', markersize=8, alpha=0.8)
                else:
                    # No data available - plot a black X at y=20
                    ax.scatter(
                        [position + offset], 
                        [20], 
                        marker='x', 
                        color='black',
                        s=100,  # Slightly larger for visibility
                        linewidth=3,
                        zorder=3
                    )
                
                # Always increment offset for proper spacing
                offset += 0.45
        
        # Calculate centered tick positions and sample sizes per dataset
        centered_positions = []
        method_sample_sizes = {}
        
        for method_name in methods:
            base_position = method_positions[method_name]
            n_datasets = 0
            dataset_samples = []
            
            for dataset_key in sorted_dataset_names:
                if dataset_key not in filtered_time_data:
                    continue
                dataset_data = filtered_time_data[dataset_key]
                found_method = False
                for method_key, data in dataset_data.items():
                    key_method = method_key.split('/')[-1]
                    key_method = key_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                    key_method = key_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                    key_method = key_method.replace('-preview-06-17', '')
                    key_method = key_method.replace('-20250805', '')
                    if key_method == method_name:
                        n_datasets += 1
                        dataset_samples.append(data['n_runs'])
                        found_method = True
                        break
                
                # If method not found in this dataset, add 0 to show no successful runs
                if not found_method:
                    n_datasets += 1
                    dataset_samples.append(0)
            
            center_offset = (n_datasets - 1) * 0.45 / 2
            centered_positions.append(base_position + center_offset)
            method_sample_sizes[method_name] = dataset_samples
        
        # Add vertical separator lines between methods
        if len(methods) > 1:
            for i in range(len(methods) - 1):
                separator_x = (centered_positions[i] + centered_positions[i + 1]) / 2
                ax.axvline(x=separator_x, color='gray', linestyle='-', linewidth=2, alpha=0.7, zorder=0)
        
        # Configure the plot
        ax.set_xticks(centered_positions)
        
        # Handle method names with special formatting and add sample sizes
        xticklabels = []
        for method in methods:
            # Format method name
            if '-thinking' in method or '-medium' in method:
                method_label = "-".join(method.split('-')[:-1]) + "\n" + method.split('-')[-1]
            else:
                method_label = method
            
            # Add sample size information as list
            if method in method_sample_sizes:
                dataset_samples = method_sample_sizes[method]
                if dataset_samples:
                    sample_list = ','.join(map(str, dataset_samples))
                    method_label += f"\n(n=[{sample_list}])"
            
            xticklabels.append(method_label)
        
        ax.set_xticklabels(xticklabels, rotation=0, ha='center', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax.grid(False, axis='x')
        ax.set_ylim(-0.5, 20.5)  # Similar to figure_6.py
        ax.set_yticks(list(range(0, 21, 2)))  # Whole numbers, similar to figure_6.py
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=12)  # Smaller font for longer labels
        ax.set_ylabel('Iteration to Reach Threshold', fontsize=16, fontweight='bold')
        
        # Make plot border black
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(save_path, exist_ok=True)
        if threshold_percent == 0.8:
            figure_prefix = 'figure_S4'
        else:
            figure_prefix = 'figure_S5'
        plt.savefig(os.path.join(save_path, f'{figure_prefix}_{provider_name}.png'), 
                    dpi=300, bbox_inches='tight')
        # plt.show()
    
    return fig

if __name__ == "__main__":
    # Get run path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')
    
    bo_benchmark_path = os.path.join(run_path, 'bayesian/{dataset_name}/benchmark/')
    llm_benchmark_path = os.path.join(run_path, 'llm/{dataset_name}/benchmark/')
    random_path = os.path.join(run_path, 'random/{dataset_name}/')
    n_tracks = 20
    track_size = 20

    path_dict = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        bo_path = bo_benchmark_path.format(dataset_name=dataset_name)
        llm_path = llm_benchmark_path.format(dataset_name=dataset_name)
        random_dataset_path = random_path.format(dataset_name=dataset_name)
        
        # Process BO paths
        if os.path.exists(bo_path):
            bo_paths = [os.path.join(bo_path, path) for path in os.listdir(bo_path)]
            for path in bo_paths:
                if path.endswith('-20') or path.endswith('-20-des0'):
                    track_data, run_dirs = get_tracks(path, dataset_name, bo=True, n_tracks=n_tracks, track_size=track_size, return_rundir=True)
                    if track_data is not None:
                        path_dict[dataset_name][path] = track_data
        
        # Process LLM paths
        if os.path.exists(llm_path):
            llm_paths = [os.path.join(llm_path, path) for path in os.listdir(llm_path)]
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
        
        # Process Random runs
        if os.path.exists(random_dataset_path):
            try:
                track_data, run_dirs = get_tracks(random_dataset_path, dataset_name, n_tracks=n_tracks, track_size=track_size, return_rundir=True)
                if track_data is not None:
                    random_key = os.path.join(random_dataset_path, 'random')
                    path_dict[dataset_name][random_key] = track_data
            except Exception as e:
                print(f'Error processing random runs for {dataset_name}: {e}')

    print(f"\nDataset summary:")
    for dataset_name, paths in path_dict.items():
        print(f"{dataset_name}: {len(paths)} method(s)")

    # Create time-to-best plots for both 80% and 95% thresholds
    print("\n=== Creating Time-to-80% Plots ===")
    time_80_data = get_time_to_best_data(path_dict, threshold_percent=0.8)
    create_time_to_best_plot(time_80_data, 0.8)
    
    print("\n=== Creating Time-to-95% Plots ===")
    time_95_data = get_time_to_best_data(path_dict, threshold_percent=0.95)
    create_time_to_best_plot(time_95_data, 0.95)
    
    print("\nTime-to-best analysis complete!")

    print(f'Figure S4 Anthropic saved to ./pngs/figure_S4_Anthropic.png')
    print(f'Figure S4 Google saved to ./pngs/figure_S4_Google.png')
    print(f'Figure S4 OpenAI saved to ./pngs/figure_S4_OpenAI.png')
    print(f'Figure S4 Atlas saved to ./pngs/figure_S4_Atlas.png')
    print(f'Figure S5 Anthropic saved to ./pngs/figure_S5_Anthropic.png')
    print(f'Figure S5 Google saved to ./pngs/figure_S5_Google.png')
    print(f'Figure S5 OpenAI saved to ./pngs/figure_S5_OpenAI.png')
    print(f'Figure S5 Atlas saved to ./pngs/figure_S5_Atlas.png')
