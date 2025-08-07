import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import glob, os
import sys
from olympus.datasets.dataset import Dataset

# Set font to sf pro
plt.rcParams['font.family'] = 'SF Pro Display'

dataset_names = [
    'Buchwald_Hartwig', 
    'Suzuki_Doyle', 
    'Suzuki_Cernak', 
    'Reductive_Amination', 
    'Alkylation_Deprotection', 
    'Chan_Lam_Full'
]

dataset_to_obj = {
    'Buchwald_Hartwig': 'yield',
    'Suzuki_Doyle': 'yield', 
    'Suzuki_Cernak': 'conversion',
    'Reductive_Amination': 'percent_conversion',
    'Alkylation_Deprotection': 'yield',
    'Chan_Lam_Full': {
        'objectives': ['desired_yield', 'undesired_yield'],
        'transform': 'subtract',  # desired - undesired
        'order': [0, 1],  # subtract objectives[1] from objectives[0]
        'aggregation': 'mean'
    }
}

model_to_provider = {
    "Anthropic": [
        'claude-3-5-haiku',
        'claude-3-5-sonnet',
        'claude-3-7-sonnet',
        'claude-3-7-sonnet-thinking',
        'claude-sonnet-4',
        'claude-opus-4',
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
        'gpt-4o-mini',
        'gpt-4o',
        'gpt-4.1-mini',
        'gpt-4.1',
        'o4-mini-low',
        'o3-low',
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
    'reductive_amination': '#221150',
    'buchwald_hartwig': '#5e177f',
    'chan_lam_full': '#972c7f',
    'suzuki_cernak': '#d3426d',
    'suzuki_doyle': '#f8755c',
    'alkylation_deprotection': '#febb80'
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
                
                # Process groups and extract objectives
                for name, group in param_groups:
                    try:
                        obj_value = get_objective_value(group, dataset_config, aggregation)
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

def get_top_obs_data(path_dict):
    # Extract the top observation for each track for each model across all datasets
    top_obs_data = {}
    for dataset_name, optimization_data in path_dict.items():
        top_obs_data[dataset_name.lower()] = {} 
        for key, tracks in optimization_data.items():
            top_obs_data[dataset_name.lower()][key] = {'top_obs': [], 'median': None, 'q1': None, 'q3': None}
            # Get the max in each row
            try:
                top_obs = [float(np.max(track)) for track in tracks]
                top_obs_data[dataset_name.lower()][key]['top_obs'] = top_obs
            
                median = float(np.median(top_obs))
                q1 = float(np.percentile(top_obs, 25))
                q3 = float(np.percentile(top_obs, 75))
                top_obs_data[dataset_name.lower()][key]['median'] = median
                top_obs_data[dataset_name.lower()][key]['q1'] = q1
                top_obs_data[dataset_name.lower()][key]['q3'] = q3
            except Exception as e:
                print(f'Error getting top obs for {key} in {dataset_name}: {e}')
    return top_obs_data

def load_optimization_data(selection_path_dict, dataset_name):
    """
    Load and organize optimization data from the selection_path_dict structure
    """
    optimization_data = {}
    
    if dataset_name not in selection_path_dict:
        print(f"Dataset {dataset_name} not found in selection_path_dict")
        return optimization_data
    
    for method_path, runs in selection_path_dict[dataset_name].items():
        method_name = method_path.split('/')[-1] if '/' in method_path else method_path
        optimization_data[method_name] = []
        
        for run_path, selections in runs.items():
            run_data = []
            for selection in selections:
                # Convert numpy arrays to lists for easier handling
                selection_dict = {}
                for param, values in selection.items():
                    if hasattr(values, 'tolist'):
                        selection_dict[param] = values.tolist()
                    else:
                        selection_dict[param] = values
                run_data.append(selection_dict)
            optimization_data[method_name].append(run_data)
    
    return optimization_data

def compute_parameter_selection_probabilities(optimization_data, param_options):
    """
    Compute probability of each parameter option being selected at each index
    
    Args:
        optimization_data: Dictionary with method names as keys and list of runs as values
        param_options: Dictionary with parameter names as keys and list of options as values
    
    Returns:
        Dictionary with probabilities for each method and parameter
    """
    probabilities = {}
    
    for method_name, runs in optimization_data.items():
        probabilities[method_name] = {}
        
        for param_name, options in param_options.items():
            probabilities[method_name][param_name] = {}
            
            # Find maximum track length across all runs
            max_length = 0
            for run in runs:
                max_length = max(max_length, len(run))
            
            # Initialize probability arrays
            for option in options:
                probabilities[method_name][param_name][option] = np.zeros(max_length)
            
            # Count selections at each index
            counts = defaultdict(lambda: defaultdict(int))
            total_counts = defaultdict(int)
            
            for run in runs:
                for idx, selection in enumerate(run):
                    if param_name in selection:
                        param_values = selection[param_name]
                        # Handle case where param_values might be a list or single value
                        if isinstance(param_values, list):
                            for value in param_values:
                                if value in options:
                                    counts[idx][value] += 1
                                    total_counts[idx] += 1
                        else:
                            if param_values in options:
                                counts[idx][param_values] += 1
                                total_counts[idx] += 1
            
            # Compute probabilities
            for idx in range(max_length):
                if total_counts[idx] > 0:
                    for option in options:
                        probabilities[method_name][param_name][option][idx] = counts[idx][option] / total_counts[idx]
    
    return probabilities

def plot_parameter_selection_probabilities(probabilities, param_options, save_path=None):
    """
    Create visualization of parameter selection probabilities
    """
    methods = list(probabilities.keys())
    params = list(param_options.keys())
    
    # Create subplots for each parameter
    n_params = len(params)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_params, n_methods, figsize=(5*n_methods, 4*n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, max([len(options) for options in param_options.values()])))
    
    for i, param_name in enumerate(params):
        options = param_options[param_name]
        
        for j, method_name in enumerate(methods):
            ax = axes[i, j]
            
            if param_name in probabilities[method_name]:
                # Plot probability curves for each option
                max_length = max([len(probs) for probs in probabilities[method_name][param_name].values()])
                x_indices = range(max_length)
                
                for k, option in enumerate(options):
                    probs = probabilities[method_name][param_name][option]
                    # Only plot non-zero parts
                    nonzero_indices = np.nonzero(probs)[0]
                    if len(nonzero_indices) > 0:
                        ax.plot(nonzero_indices, probs[nonzero_indices], 
                               label=f"opt_{k}",
                               color=colors[k], marker='o', markersize=3)
                
                ax.set_title(f'{method_name}\n{param_name}')
                ax.set_xlabel('Optimization Step')
                ax.set_ylabel('Selection Probability')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method_name}\n{param_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_heatmap_analysis(probabilities, param_options, save_path=None):
    """
    Create heatmap visualization showing parameter selection patterns
    """
    methods = list(probabilities.keys())
    
    for method_name in methods:
        n_params = len(param_options)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, options) in enumerate(param_options.items()):
            if param_name in probabilities[method_name]:
                # Create probability matrix
                max_length = max([len(probs) for probs in probabilities[method_name][param_name].values()])
                prob_matrix = np.zeros((len(options), max_length))
                
                for j, option in enumerate(options):
                    probs = probabilities[method_name][param_name][option]
                    prob_matrix[j, :len(probs)] = probs
                
                # Create heatmap
                im = axes[i].imshow(prob_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
                
                # Set labels
                axes[i].set_yticks(range(len(options)))
                axes[i].set_yticklabels([f"opt_{j}" for j in range(len(options))])
                axes[i].set_xlabel('Optimization Step')
                axes[i].set_title(f'{param_name} - {method_name}')

                axes[i].set_xticks(range(0, 20, 1))
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], label='Selection Probability')

            plt.tight_layout()
        
        if save_path:
            method_save_path = save_path.replace('.png', f'_{method_name}_heatmap.png')
            plt.savefig(method_save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def analyze_selection_diversity(probabilities, param_options):
    """
    Analyze the diversity of parameter selections across optimization steps
    """
    diversity_results = {}
    
    for method_name, method_data in probabilities.items():
        diversity_results[method_name] = {}
        
        for param_name, param_data in method_data.items():
            # Calculate Shannon entropy at each step
            max_length = max([len(probs) for probs in param_data.values()])
            entropies = []
            
            for step in range(max_length):
                step_probs = []
                for option in param_options[param_name]:
                    if step < len(param_data[option]):
                        step_probs.append(param_data[option][step])
                    else:
                        step_probs.append(0)
                
                # Calculate Shannon entropy
                step_probs = np.array(step_probs)
                step_probs = step_probs[step_probs > 0]  # Remove zeros
                if len(step_probs) > 0:
                    entropy = -np.sum(step_probs * np.log2(step_probs))
                    entropies.append(entropy)
                else:
                    entropies.append(0)
            
            diversity_results[method_name][param_name] = entropies
    
    return diversity_results

def plot_diversity_analysis(diversity_results, save_path=None):
    """
    Plot diversity analysis results
    """
    methods = list(diversity_results.keys())
    params = list(diversity_results[methods[0]].keys())
    
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 3*len(params)))
    if len(params) == 1:
        axes = [axes]
    
    for i, param_name in enumerate(params):
        for method_name in methods:
            if param_name in diversity_results[method_name]:
                entropies = diversity_results[method_name][param_name]
                axes[i].plot(range(len(entropies)), entropies, 
                           label=method_name, marker='o', markersize=3)
        
        axes[i].set_title(f'Selection Diversity - {param_name}')
        axes[i].set_xlabel('Optimization Step')
        axes[i].set_ylabel('Shannon Entropy')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(0, 20, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_entropy(probabilities):
    formatted = {}
    for param_name, data in probabilities.items():
        formatted[param_name] = {}
        for option, probs in data.items():
            for idx, prob in enumerate(probs):
                if idx+1 not in formatted[param_name]:
                    formatted[param_name][idx+1] = []
                formatted[param_name][idx+1].append(float(prob))
    
    entropy_results = {}
    for param_name, iteration_data in formatted.items():
        entropy_results[param_name] = []
        for idx in iteration_data.keys():
            probs = np.asarray(iteration_data[idx])
            n_outcomes = len(probs)
            max_entropy = np.log2(n_outcomes) if n_outcomes > 1 else 1
            entropy = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0)) / max_entropy
            entropy_results[param_name].append(float(entropy))
    return entropy_results

def compute_average_entropy(entropy_data):
    """Compute average entropy across all parameters for each iteration"""
    if not entropy_data:
        return []
    
    # Get the maximum length across all parameters
    max_length = max(len(param_data) for param_data in entropy_data.values())
    
    # Initialize array to store average entropies
    avg_entropies = []
    
    for iteration in range(max_length):
        iteration_entropies = []
        for param_name, param_data in entropy_data.items():
            if iteration < len(param_data):
                iteration_entropies.append(param_data[iteration])
        
        if iteration_entropies:
            avg_entropies.append(np.mean(iteration_entropies))
        else:
            avg_entropies.append(0)
    
    return avg_entropies

def create_provider_entropy_plot(provider_name, provider_methods, entropies, datasets_to_run, dataset_to_color):
    """
    Create entropy plots for a given provider with subplots for each method
    Each subplot shows entropy traces for that method across all datasets
    """
    
    # Filter methods that have entropy data
    available_methods = []
    for method in provider_methods:
        method_found = False
        for dataset_name in datasets_to_run:
            if dataset_name in entropies and any(method in key for key in entropies[dataset_name].keys()):
                method_found = True
                break
        if method_found or '-des' in method or 'claude-3-7-sonnet-thinking' in method:
            available_methods.append(method)
        else:
            print(f'{method} not found for {provider_name}')
    
    if not available_methods:
        print(f"No entropy data found for {provider_name}")
        return None
    
    # Calculate subplot layout
    n_methods = len(available_methods)
    if n_methods <= 3:
        rows, cols = 1, n_methods
    elif n_methods <= 6:
        rows, cols = 2, 3
    elif n_methods <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, (n_methods + 3) // 4
    
    # Create figure
    fig_width = max(5 * cols, 15)
    fig_height = max(4 * rows, 8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is always 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each method
    for method_idx, method_name in enumerate(available_methods):
        row = method_idx // cols
        col = method_idx % cols
        ax = axes[row, col]
        
        # Plot entropy traces for this method across all datasets
        for dataset_name in datasets_to_run:
            if dataset_name in entropies:
                dataset_color = dataset_to_color.get(dataset_name.lower(), '#1f77b4')
                
                # Find the matching method in the entropy data
                method_data = None
                for key, entropy_data in entropies[dataset_name].items():
                    # Clean the key to match method names
                    clean_key = key.replace('-1-20-20', '').replace('-latest', '')
                    clean_key = clean_key.replace('-preview-03-25', '').replace('-preview-04-17', '')
                    clean_key = clean_key.replace('-20250514', '').replace('-des0', '-des')
                    clean_key = clean_key.replace('-preview-06-17', '')
                    
                    if clean_key == method_name:
                        method_data = entropy_data
                        break
                
                if method_data is not None:
                    # Average entropy across all parameters for this method-dataset combination
                    avg_entropy = compute_average_entropy(method_data)
                    
                    if avg_entropy:
                        iterations = range(1, len(avg_entropy) + 1)
                        ax.plot(iterations, avg_entropy,
                               color=dataset_color,
                               linewidth=2.5,
                               alpha=0.8,
                               label=dataset_name.replace('_', ' ').title())
        
        # Customize subplot
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Normalized Entropy', fontsize=12)
        ax.set_title(f'{method_name}', fontsize=16)
        ax.set_xlim(0.5, 21)
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.grid(False, axis='x')
        
        # # Add legend to first subplot only
        # if method_idx == 0:
        #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Hide empty subplots
    total_subplots = rows * cols
    for idx in range(n_methods, total_subplots):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Add main title
    # fig.suptitle(f'{provider_name} - Parameter Selection Entropy by Method', 
    #              fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.85)
    plt.show()
    
    return fig

def create_provider_entropy_plot_gridspec(provider_name, provider_methods, entropies, datasets_to_run, dataset_to_color):
    """
    Create entropy plots using gridspec for better layout control
    """
    # Filter methods that have entropy data
    available_methods = []
    for method in provider_methods:
        method_found = False
        for dataset_name in datasets_to_run:
            if dataset_name in entropies and any(method in key for key in entropies[dataset_name].keys()):
                method_found = True
                break
        if method_found or '-des' in method or 'claude-3-7-sonnet-thinking' in method:
            available_methods.append(method)
    
    if not available_methods:
        print(f"No entropy data found for {provider_name}")
        return None
    
    n_methods = len(available_methods)
    
    # Create figure and gridspec
    fig = plt.figure(figsize=(15, 8))
    
    if n_methods == 5:
        # Create 2x3 grid but use custom positioning
        gs = gridspec.GridSpec(2, 6, figure=fig)
        
        # Top row - 3 subplots spanning 2 columns each
        subplot_positions = [
            gs[0, 0:2],  # Top left
            gs[0, 2:4],  # Top center  
            gs[0, 4:6],  # Top right
            gs[1, 1:3],  # Bottom left (offset by 1 column)
            gs[1, 3:5],  # Bottom right (offset by 1 column)
        ]
    else:
        # Default gridspec behavior
        if n_methods <= 3:
            rows, cols = 1, n_methods
        elif n_methods <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        subplot_positions = [gs[i // cols, i % cols] for i in range(n_methods)]
    
    # Create subplots
    axes = []
    for method_idx, method_name in enumerate(available_methods):
        ax = fig.add_subplot(subplot_positions[method_idx])
        axes.append(ax)
        
        # Plot entropy traces for this method across all datasets
        for dataset_name in datasets_to_run:
            if dataset_name in entropies:
                dataset_color = dataset_to_color.get(dataset_name.lower(), '#1f77b4')
                
                # Find the matching method in the entropy data
                method_data = None
                for key, entropy_data in entropies[dataset_name].items():
                    clean_key = key.replace('-1-20-20', '').replace('-latest', '')
                    clean_key = clean_key.replace('-preview-03-25', '').replace('-preview-04-17', '')
                    clean_key = clean_key.replace('-20250514', '').replace('-des0', '-des')
                    clean_key = clean_key.replace('-preview-06-17', '')
                    
                    if clean_key == method_name:
                        method_data = entropy_data
                        break
                
                if method_data is not None:
                    avg_entropy = compute_average_entropy(method_data)
                    
                    if avg_entropy:
                        iterations = range(1, len(avg_entropy) + 1)
                        ax.plot(iterations, avg_entropy,
                               color=dataset_color,
                               linewidth=2.5,
                               alpha=0.8,
                               label=dataset_name.replace('_', ' ').title(),
                               marker='o',
                               markersize=4,
                               markerfacecolor='white')
        
        # Customize subplot
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Normalized Entropy\nAcross Parameters', fontsize=12)
        ax.set_title(f'{method_name}', fontsize=16)
        ax.set_xlim(0.5, 21)
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.grid(False, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return fig

import matplotlib.pyplot as plt
import numpy as np
import math

def clean_method_name(method):
    """Clean method name to match ALL_METHODS using your original cleaning rules"""
    clean_method = method.replace('-1-20-20', '').replace('-latest', '')
    clean_method = clean_method.replace('-preview-03-25', '').replace('-preview-04-17', '')
    clean_method = clean_method.replace('-20250514', '').replace('-des0', '-des')
    clean_method = clean_method.replace('-preview-06-17', '')
    return clean_method

def get_provider_color(method, provider_models):
    """Get color based on provider"""
    colors = {'Anthropic': 'blue', 'Google': 'red', 'OpenAI': 'green', 'Atlas': 'orange'}
    
    for provider, models in provider_models.items():
        for model in models:
            if model in method:
                return colors.get(provider, 'gray')
    return 'gray'

def create_entropy_figures(entropies, provider_models):
    """Create figures showing entropy evolution for each dataset"""
    ALL_METHODS = [item for sublist in provider_models.values() for item in sublist]
    dataset_figures = {}
    
    for dataset_name, dataset_data in entropies.items():
        print(f"Processing {dataset_name}...")
        
        # Filter and clean method names
        valid_methods = {}
        for method, param_data in dataset_data.items():
            clean_method = clean_method_name(method)
            if clean_method in ALL_METHODS:
                # Only include methods that have non-zero data
                if any(any(values) for values in param_data.values()):
                    valid_methods[clean_method] = param_data
        
        if not valid_methods:
            print(f"No valid methods found for {dataset_name}")
            continue
        
        # Sort methods according to ALL_METHODS order
        ordered_methods = [(method, valid_methods[method]) for method in ALL_METHODS if method in valid_methods]
        
        # Calculate grid dimensions
        n_methods = len(ordered_methods)
        n_cols = min(6, n_methods)  # Maximum 6 columns
        n_rows = math.ceil(n_methods / n_cols)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'\n{dataset_name} - Entropy Evolution by Method\n', fontsize=24)
        
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
            
            # Plot each parameter with consistent colors
            for param_name, values in param_data.items():
                # if values and any(v != 0 for v in values):  # Only plot if there are non-zero values
                iterations = [i+1 for i in range(len(values))]
                ax.plot(iterations, values, 
                        marker='o', markersize=4, linewidth=2, 
                        color=param_colors[param_name], 
                        label=param_name, alpha=0.8)
            
            ax.set_title(f'{method}', fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Entropy')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1, 21)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)
        
        # Create a single legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.08), ncol=len(labels), fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the legend below
        dataset_figures[dataset_name] = fig
        
        print(f"Created figure for {dataset_name} with {n_methods} methods")
    
    return dataset_figures

def create_entropy_figures_detailed(entropies, provider_models):
    """Create detailed figures with each method-parameter as separate subplot"""
    ALL_METHODS = [item for sublist in provider_models.values() for item in sublist]
    dataset_figures = {}
    
    for dataset_name, dataset_data in entropies.items():
        print(f"Processing {dataset_name} (detailed)...")
        
        # Collect all method-parameter combinations in ALL_METHODS order
        method_param_combinations = []
        for method in ALL_METHODS:
            # Find the original method name that maps to this clean method
            for original_method, param_data in dataset_data.items():
                clean_method = clean_method_name(original_method)
                if clean_method == method:
                    for param_name, values in param_data.items():
                        if values and any(v != 0 for v in values):  # Only include non-zero data
                            method_param_combinations.append((clean_method, param_name, values))
                    break  # Move to next method in ALL_METHODS
        
        if not method_param_combinations:
            continue
        
        # Calculate grid dimensions
        n_plots = len(method_param_combinations)
        n_cols = min(8, n_plots)  # Maximum 8 columns for readability
        n_rows = math.ceil(n_plots / n_cols)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        fig.suptitle(f'{dataset_name} - Entropy Evolution (Method-Parameter Detail)', fontsize=16)
        
        # Handle subplot indexing
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each method-parameter combination
        for idx, (method, param_name, values) in enumerate(method_param_combinations):
            ax = axes[idx]
            
            provider_color = get_provider_color(method, provider_models)
            iterations = range(len(values))
            
            ax.plot(iterations, values, marker='o', markersize=4, 
                   linewidth=2, color=provider_color, alpha=0.8)
            ax.set_title(f'{method}\n{param_name}', fontsize=10)
            ax.set_xlabel('Iteration', fontsize=8)
            ax.set_ylabel('Entropy', fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        dataset_figures[dataset_name] = fig
        
        print(f"Created detailed figure for {dataset_name} with {n_plots} method-parameter combinations")
    
    return dataset_figures

if __name__ == "__main__":
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

    top_obs_data = get_top_obs_data(path_dict)
    sorted_dataset_names = sorted(dataset_to_color.keys(), key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), reverse=True)
    top_obs_data = {k: top_obs_data[k] for k in sorted_dataset_names}

        # Group methods by provider for boxplots
    method_data = {}
    method_names = []
    # FILTER_METHODS = False  # Set to False to include all methods
    remove_datasets = []

    # First collect all method names and their dataset coverage
    for dataset_name, dataset_data in top_obs_data.items():
        for key in dataset_data.keys():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '')
            method_name = method_name.replace('-latest', '')
            method_name = method_name.replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '')
            method_name = method_name.replace('-preview-04-17', '')
            method_name = method_name.replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            if 'gpt-4.1' in method_name and ('nano' in method_name):
                continue
            method_names.append(method_name)

    # Remove duplicates while preserving order
    method_names = list(dict.fromkeys(method_names))

    dataset_names = list(path_dict.keys())
    dataset_OHE = {}
    dataset_param_options = {}
    for dataset_name in dataset_names:
        OHE_lookup = {}
        expected_size = 0
        param_names = []
        param_options = {}
        dataset = Dataset(dataset_name)
        for i, p in enumerate(dataset.param_space):
            name = p.name
            options = p.options
            OHE_lookup[name] = {o: [1 if i==j else 0 for j in range(len(options))] for i,o in enumerate(options)}
            expected_size += len(options)
            param_names.append(name)
            param_options[name] = options
        dataset_OHE[dataset_name] = OHE_lookup
        dataset_param_options[dataset_name] = param_options

    selection_path_dict = {}
    for dataset_name, optimization_data in path_dict.items():
        if dataset_name not in dataset_names:
            continue
        selection_path_dict[dataset_name] = {}
        objectives = dataset_to_obj[dataset_name]
        if isinstance(objectives, dict):
            objectives = objectives["objectives"]
        else:
            objectives = [objectives]
        param_names = list(dataset_param_options[dataset_name].keys())
        print(f'--- {dataset_name} ---')
        for method_path, _ in optimization_data.items():
            # print(f'   --- {method_path} ---')
            runs = glob.glob(method_path + "/run_*/")
            run_dict = {}
            for run in runs:
                with open(run + "seen_observations.json", "r") as f:
                    seen_observations = json.load(f)

                df = pd.DataFrame(seen_observations)
                
                # Group the observations by the "reasoning" if "llm" in method_path
                if "llm" in method_path:
                    group_cols = ["reasoning", "explanation"]
                    remove_cols = []
                # The observations are a list of dictionaries
                elif "bayesian" in method_path:
                    all_cols = df.columns.tolist()
                    group_cols = [c for c in all_cols if c not in objectives]
                
                grouped_dfs = {name: group for name, group in df.groupby(group_cols, sort=False)}
                grouped_list = list(df.groupby(group_cols, sort=False))
                selections = []
                for group in grouped_list:
                    name = group[0]
                    group_df = group[1]
                    selection_dict = {k: None for k in param_names}
                    for col in group_df.columns:
                        if col in param_names:
                            all_values = group_df[col].tolist()
                            selection_dict[col] = all_values[0]
                    selections.append(selection_dict)
                
                if "bayesian" in method_path:
                    selections = selections[1:]
                run_dict[run] = selections
            selection_path_dict[dataset_name][method_path] = run_dict
        
    dataset_probabilities = {}
    for dataset_name in dataset_names:
        param_options = dataset_param_options[dataset_name]
        optimization_data = load_optimization_data(
            selection_path_dict, 
            dataset_name
            )
        probabilities = compute_parameter_selection_probabilities(
            optimization_data, 
            param_options
        )
        dataset_probabilities[dataset_name] = {
            "optimization_data": optimization_data, 
            "probabilities": probabilities
        }

    entropies = {}
    for dataset_name in dataset_names:
        probabilities = dataset_probabilities[dataset_name]['probabilities']
        entropies[dataset_name] = {}
        print(f'--- {dataset_name} ---')
        for k in probabilities.keys():
            entropy = compute_entropy(probabilities[k])
            entropies[dataset_name][k] = entropy

    # Create plots for each provider
    print("Creating entropy plots for each provider...")

    # Group methods by provider
    provider_methods = {}
    for provider, models_list in model_to_provider.items():
        # Only include methods that are in our filtered method_names list
        provider_filtered_methods = [method for method in models_list if method in method_names]
        if provider_filtered_methods:  # Only include providers that have methods
            provider_methods[provider] = provider_filtered_methods
        else:
            print(f'No methods found for provider: {provider}')

    for provider_name, methods in provider_methods.items():
        print(f"\nCreating entropy plot for {provider_name}...")
        fig = create_provider_entropy_plot_gridspec(
            provider_name, 
            methods, 
            entropies, 
            dataset_names, 
            dataset_to_color
        )
        
        if fig is not None:
            # Save the figure to ./pngs/figure_7_{provider_name}.png
            os.makedirs('./pngs', exist_ok=True)
            plt.savefig(f'./pngs/figure_7_{provider_name}.png', dpi=300, bbox_inches='tight')

    print("\nCompleted entropy analysis plots for all providers.")

    provider_models = {
        'Anthropic': [
            'claude-3-5-haiku',
            'claude-3-5-sonnet',
            'claude-3-7-sonnet',
            'claude-3-7-sonnet-thinking',
            'claude-sonnet-4',
            'claude-opus-4'
        ],
        'Google': [
            'gemini-2.0-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.5-flash',
            'gemini-2.5-flash-medium',
            'gemini-2.5-pro',
            'gemini-2.5-pro-medium'
        ],
        'OpenAI': [
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4.1-mini',
            'gpt-4.1',
            'o4-mini-low',
            'o3-low'
        ],
        'Atlas': [
            'atlas-ei',
            'atlas-ei-des',
            'atlas-ucb',
            'atlas-ucb-des',
            'atlas-pi',
            'atlas-pi-des'
        ]
    }

    dataset_figures = create_entropy_figures(entropies, provider_models)

    for dataset_name, fig in dataset_figures.items():
        # Save the figure to ./pngs/figure_SI_entropy_analysis_{dataset_name}.png
        os.makedirs('./pngs', exist_ok=True)
        plt.savefig(f'./pngs/figure_SI_entropy_analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')