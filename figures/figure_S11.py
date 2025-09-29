import pandas as pd
import glob, json, os
import numpy as np
import sys

import matplotlib.pyplot as plt

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
    Returns NaN if any required objective is missing or invalid
    """
    if isinstance(dataset_config, str):
        # Simple single objective
        if hasattr(obs_or_group, 'iloc'):  # It's a pandas group
            if aggregation == 'first':
                value = obs_or_group[dataset_config].iloc[0]
            elif aggregation == 'mean':
                value = obs_or_group[dataset_config].mean()
            elif aggregation == 'max':
                value = obs_or_group[dataset_config].max()
            elif aggregation == 'min':
                value = obs_or_group[dataset_config].min()
            elif aggregation == 'median':
                value = obs_or_group[dataset_config].median()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Single observation
            if dataset_config not in obs_or_group:
                return np.nan
            value = obs_or_group[dataset_config]
        
        # Check if value is valid
        return value if pd.notna(value) else np.nan
    else:
        # Complex multi-objective
        if hasattr(obs_or_group, 'iloc'):  # It's a pandas group
            objectives = []
            for obj_name in dataset_config['objectives']:
                if obj_name not in obs_or_group.columns:
                    return np.nan
                    
                if aggregation == 'first':
                    obj_val = obs_or_group[obj_name].iloc[0]
                elif aggregation == 'mean':
                    obj_val = obs_or_group[obj_name].mean()
                elif aggregation == 'max':
                    obj_val = obs_or_group[obj_name].max()
                elif aggregation == 'min':
                    obj_val = obs_or_group[obj_name].min()
                elif aggregation == 'median':
                    obj_val = obs_or_group[obj_name].median()
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
                
                if pd.isna(obj_val):
                    return np.nan
                objectives.append(obj_val)
        else:
            # Single observation
            objectives = []
            for obj_name in dataset_config['objectives']:
                if obj_name not in obs_or_group:
                    return np.nan
                obj_val = obs_or_group[obj_name]
                if pd.isna(obj_val):
                    return np.nan
                objectives.append(obj_val)
        
        # Apply transformation
        if dataset_config['transform'] == 'subtract':
            order = dataset_config['order']
            result = objectives[order[0]] - objectives[order[1]]
        else:
            raise ValueError(f"Unknown transformation: {dataset_config['transform']}")
        
        return result if pd.notna(result) else np.nan

def calculate_invalid_suggestion_rates(path_dict):
    """Calculate the rate of invalid suggestions (NaN objectives) for each method across all datasets"""
    invalid_rates_data = {}
    
    for dataset_name, optimization_data in path_dict.items():
        invalid_rates_data[dataset_name.lower()] = {}
        dataset_config = dataset_to_obj[dataset_name]
        
        for method_key, run_dirs in optimization_data.items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
            run_invalid_rates = []
            
            for run_dir in run_dirs:
                seen_path = os.path.join(run_dir, 'seen_observations.json')
                
                if not os.path.exists(seen_path):
                    continue
                
                with open(seen_path, 'r') as f:
                    seen_data = json.load(f)
                
                total_suggestions = 0
                invalid_suggestions = 0
                
                if dataset_name == 'Chan_Lam_Full':
                    # Special handling for Chan_Lam_Full
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
                    
                    # Process groups and check for invalid objectives
                    for name, group in param_groups:
                        total_suggestions += 1
                        try:
                            obj_value = get_objective_value(group, dataset_config, 'mean')
                            if pd.isna(obj_value):
                                invalid_suggestions += 1
                        except (KeyError, TypeError, ZeroDivisionError):
                            invalid_suggestions += 1
                else:
                    # Standard handling for other datasets
                    for obs in seen_data:
                        total_suggestions += 1
                        try:
                            obj_value = get_objective_value(obs, dataset_config)
                            if pd.isna(obj_value):
                                invalid_suggestions += 1
                        except (KeyError, TypeError, ZeroDivisionError):
                            invalid_suggestions += 1
                
                # Store absolute count of invalid suggestions
                run_invalid_rates.append(invalid_suggestions)
            
            if run_invalid_rates:
                invalid_rates_data[dataset_name.lower()][method_key] = {
                    'invalid_rates': run_invalid_rates,
                    'median': np.median(run_invalid_rates),
                    'q1': np.percentile(run_invalid_rates, 25),
                    'q3': np.percentile(run_invalid_rates, 75)
                }
    
    return invalid_rates_data

def create_individual_provider_invalid_plot(provider_name, provider_method_list, invalid_rates_data, remove_datasets=[]):
    """Create a single plot for one provider showing invalid suggestion rates"""
    if not provider_method_list:
        print(f"No methods found for {provider_name}")
        return
    
    invalid_rates_refined = {k: v for k, v in invalid_rates_data.items() if k not in remove_datasets}
    
    # Fixed figure sizing for consistent layout
    fig_width = 15
    fig_height = 8
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Add horizontal line at y=0 in the background
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
    
    # Create mapping of method names to their positions with better spacing
    method_positions = {name: idx * 3 + 1 for idx, name in enumerate(provider_method_list)}
    
    for method_name in provider_method_list:
        position = method_positions[method_name]
        offset = 0
        
        for dataset_name, dataset_data in invalid_rates_refined.items():
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
                        method_data['invalid_rates'],
                        positions=[position + offset],
                        widths=0.3,
                        patch_artist=True,
                        showfliers=True,
                        zorder=3
                    )
                    IQR = np.percentile(method_data['invalid_rates'], 75) - np.percentile(method_data['invalid_rates'], 25)
                    
                    # Customize boxplot appearance
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
    
    # Calculate centered tick positions
    centered_positions = []
    for method_name in provider_method_list:
        base_position = method_positions[method_name]
        # Count how many datasets have data for this method
        n_datasets = 0
        for dataset_name, dataset_data in invalid_rates_refined.items():
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
    ax.set_ylim(0, 20)  # Max 20 invalid suggestions per run
    ax.set_yticks(range(0, 21, 2))  # Y-axis ticks every 2 units: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    
    # Set y-axis label
    ax.set_ylabel('Invalid Suggestions (Count)', fontsize=18, fontweight='bold')
    
    # Make plot border black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Invalid Suggestion Rates for {provider_name} ===")
    for dataset_name in invalid_rates_refined.keys():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        for method_name in provider_method_list:
            found_data = False
            for key, method_data in invalid_rates_refined[dataset_name].items():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                current_method = current_method.replace('-preview-06-17', '')
                current_method = current_method.replace('-20250805', '')
                
                if current_method == method_name:
                    rates = method_data['invalid_rates']
                    print(f"  {method_name}: median={np.median(rates):.1f}, range=[{min(rates):.0f}-{max(rates):.0f}], n={len(rates)}")
                    found_data = True
                    break
            if not found_data:
                print(f"  {method_name}: No data")
    
    return fig, ax

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
                    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if 'run_' in d]
                    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]))
                    
                    if len(run_dirs) >= n_tracks:
                        path_dict[dataset_name][path] = run_dirs[:n_tracks]
        
        # Process LLM paths
        if os.path.exists(llm_path):
            llm_paths = [os.path.join(llm_path, path) for path in os.listdir(llm_path)]
            for path in llm_paths:
                if 'claude' in path and 'gpt' in path:
                    continue
                if '1-20-20' in path:
                    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if 'run_' in d]
                    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]))
                    
                    if len(run_dirs) >= n_tracks:
                        path_dict[dataset_name][path] = run_dirs[:n_tracks]
        
        # Process Random runs
        if os.path.exists(random_dataset_path):
            run_dirs = [os.path.join(random_dataset_path, d) for d in os.listdir(random_dataset_path) if 'run_' in d]
            run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]))
            
            if len(run_dirs) >= n_tracks:
                # Create a path key for random runs
                random_key = os.path.join(random_dataset_path, 'random')
                path_dict[dataset_name][random_key] = run_dirs[:n_tracks]

    for dataset_name, paths in path_dict.items():
        print(f"{dataset_name}: {len(paths)}")

    print("Calculating invalid suggestion rates...")
    invalid_rates_data = calculate_invalid_suggestion_rates(path_dict)

    # Sort by dataset_name based on how it appears in dataset_to_color
    sorted_dataset_names = sorted(dataset_to_color.keys(), key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), reverse=True)

    # Sort invalid_rates_data by the sorted dataset_names
    invalid_rates_data = {k: invalid_rates_data[k] for k in sorted_dataset_names if k in invalid_rates_data}
    
    # Group methods by provider for boxplots
    method_names = []
    
    # First collect all method names and their dataset coverage
    for dataset_name, dataset_data in invalid_rates_data.items():
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

    for provider_name, methods in provider_methods.items():
        print(f"\nCreating invalid suggestion rate plot for {provider_name}...")
        fig, ax = create_individual_provider_invalid_plot(provider_name, methods, invalid_rates_data, remove_datasets)
        
        # Save the figure to ./pngs/figure_invalid_suggestions_{provider_name}.png
        os.makedirs('./pngs', exist_ok=True)
        plt.savefig(f'./pngs/figure_S11_{provider_name}.png', dpi=300, bbox_inches='tight')
        
    print(f"\nCreated {len(provider_methods)} individual provider invalid suggestion plots.")

    # Print summary
    print(f"\nSummary of plotted providers:")
    for provider, methods in provider_methods.items():
        print(f"  {provider}: {len(methods)} methods - {', '.join(methods)}")
    
    # Overall statistics
    print(f"\n=== OVERALL INVALID SUGGESTION STATISTICS ===")
    
    for dataset_name in sorted_dataset_names:
        if dataset_name in invalid_rates_data:
            print(f"\n{dataset_name.replace('_', ' ').title()}:")
            
            all_rates = []
            provider_rates = {}
            
            for method_key, method_data in invalid_rates_data[dataset_name].items():
                method_name = method_key.split('/')[-1]
                method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                method_name = method_name.replace('-preview-06-17', '')
                method_name = method_name.replace('-20250805', '')
                
                # Skip filtered methods
                if 'gpt-4.1' in method_name and 'nano' in method_name:
                    continue
                
                rates = method_data['invalid_rates']
                all_rates.extend(rates)
                
                # Group by provider
                for provider, models in model_to_provider.items():
                    if method_name in models:
                        if provider not in provider_rates:
                            provider_rates[provider] = []
                        provider_rates[provider].extend(rates)
                        break
            
            if all_rates:
                print(f"  Overall: median={np.median(all_rates):.1f}, range=[{min(all_rates):.0f}-{max(all_rates):.0f}]")
                
                for provider, rates in provider_rates.items():
                    if rates:
                        print(f"  {provider}: median={np.median(rates):.1f}, range=[{min(rates):.0f}-{max(rates):.0f}], n_runs={len(rates)}")

    print(f'Figure S11 Anthropic saved to ./pngs/figure_S11_Anthropic.png')
    print(f'Figure S11 Google saved to ./pngs/figure_S11_Google.png')
    print(f'Figure S11 OpenAI saved to ./pngs/figure_S11_OpenAI.png')
    print(f'Figure S11 Atlas saved to ./pngs/figure_S11_Atlas.png')
    print(f'Figure S11 Random saved to ./pngs/figure_S11_Random.png')
