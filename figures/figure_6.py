import os
import json
import numpy as np
import matplotlib.pyplot as plt
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

def create_individual_provider_duplicate_plot(provider_name, provider_method_list, remove_datasets=[]):
    """Create a single plot for duplicate suggestions for one provider"""
    if not provider_method_list:
        print(f"No methods found for {provider_name}")
        return

    duplicate_data_refined = {k: v for k, v in duplicate_data.items() if k not in remove_datasets}

    # Fixed figure sizing for consistent layout
    fig_width = 15
    fig_height = 8

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # # Add horizontal line at y=0 in the background
    # ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)

    # Create mapping of method names to their positions with better spacing
    method_positions = {name: idx * 3 + 1 for idx, name in enumerate(provider_method_list)}

    provider_duplicate_data = {}
    for method_name in provider_method_list:
        position = method_positions[method_name]
        offset = 0

        for dataset_name, dataset_data in duplicate_data_refined.items():
            if dataset_name not in provider_duplicate_data:
                provider_duplicate_data[dataset_name] = {}
            
            for key, method_data in dataset_data.items():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '')
                current_method = current_method.replace('-latest', '')
                current_method = current_method.replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '')
                current_method = current_method.replace('-preview-04-17', '')
                current_method = current_method.replace('-des0', '-des')
                current_method = current_method.replace('-preview-06-17', '')

                if current_method == method_name:
                    provider_duplicate_data[dataset_name][method_name] = method_data['dups']
                    boxplot = ax.boxplot(
                        method_data['dups'],
                        positions=[position + offset],
                        widths=0.3,
                        patch_artist=True,
                        showfliers=True,
                        zorder=3
                    )
                    
                    IQR = np.percentile(method_data['dups'], 75) - np.percentile(method_data['dups'], 25)

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
        for dataset_name, dataset_data in duplicate_data_refined.items():
            for key in dataset_data.keys():
                current_method = key.split('/')[-1]
                current_method = current_method.replace('-1-20-20', '')
                current_method = current_method.replace('-latest', '')
                current_method = current_method.replace('-preview-03-25', '')
                current_method = current_method.replace('-20250514', '')
                current_method = current_method.replace('-preview-04-17', '')
                current_method = current_method.replace('-des0', '-des')
                current_method = current_method.replace('-preview-06-17', '')
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
    
    # Set y-axis limits appropriate for duplicate counts
    ax.set_ylim(-0.5, 20)  # Adjust based on your data range
    ax.set_yticks(list(range(0, 21, 2)))
    # ax.set_ylabel('Duplicate Suggestions', fontsize=16, labelpad=10)
    # ax.set_title(f'{provider_name} - Duplicate Suggestions Distribution', fontsize=18, fontweight='bold', pad=20)
    
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)

    # Make plot border black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, ax, provider_duplicate_data

if __name__ == "__main__":
    # Get runs path from command line argument or user input
    if len(sys.argv) > 1:
        runs_path = sys.argv[1]
        print(f"Using runs path: {runs_path}")
    else:
        runs_path = input('Enter the runs path: ')
    benchmark_path = os.path.join(runs_path, '{method}/{dataset_name}/benchmark/')

    path_dict = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        objectives = dataset_to_obj[dataset_name]
        if isinstance(objectives, dict):
            objectives = objectives['objectives']
        else:
            objectives = [objectives]
        
        for method in ['bayesian', 'llm']:
            path = benchmark_path.format(dataset_name=dataset_name, method=method)
            paths = [os.path.join(path, sub_path) for sub_path in os.listdir(path)]
            paths = [path for path in paths if path.endswith('1-20-20') or path.endswith('1-20-20-des0')]
            
            for path in paths:
                if 'atlas' in path:
                    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('/')[-1].split('_')[1]))
                    all_dups = [0 for _ in range(len(run_dirs))]
                else:
                    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    # Sort them by the last part of the name
                    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('/')[-1].split('_')[1]))
                        
                    all_dups = [] 

                    for run_dir in run_dirs:
                        # Load the seen_observations.json file
                        seen_observations_path = os.path.join(run_dir, 'seen_observations.json')
                        with open(seen_observations_path, 'r') as f:
                            seen_observations = json.load(f)

                        if dataset_name == 'Chan_Lam_Full':
                            reasoning_to_suggestion = {}
                            for obs in seen_observations:
                                reasoning = obs['reasoning']
                                if reasoning not in reasoning_to_suggestion:
                                    reasoning_to_suggestion[reasoning] = obs

                            # Remake seen_observations using the reasoning_to_suggestion map
                            seen_observations = []
                            for reasoning, suggestion in reasoning_to_suggestion.items():
                                seen_observations.append(suggestion)

                        # Get parameter keys (drop objective columns, 'reasoning', and 'explanation')
                        ks = list(seen_observations[0].keys())
                        ks_to_drop = objectives + ['reasoning', 'explanation']
                        ks_to_keep = [k for k in ks if k not in ks_to_drop]
                        suggestions = [{k: obs[k] for k in ks_to_keep} for obs in seen_observations]
                        suggestion_map = {}
                        dups = 0
                        for i, suggestion in enumerate(suggestions):
                            if suggestion not in suggestion_map.values():
                                suggestion_map[i] = suggestion
                            else:
                                dups += 1
                        all_dups.append(dups)
                        
                # Calculate statistics
                q1 = np.percentile(all_dups, 25)
                q3 = np.percentile(all_dups, 75)
                median = np.median(all_dups)
                path_dict[dataset_name][path] = {
                    'dups': all_dups,
                    "mean": np.mean(all_dups), 
                    "std": np.std(all_dups), 
                    "q1": q1, 
                    "q3": q3, 
                    "median": median
                }

    # First, let's extract the duplicate data in a similar format to the performance data
    duplicate_data = {}
    for dataset_name, dataset_paths in path_dict.items():
        duplicate_data[dataset_name.lower()] = {}
        for path, stats in dataset_paths.items():
            # Extract method name similar to performance data processing
            method_name = path.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '')
            method_name = method_name.replace('-latest', '')
            method_name = method_name.replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '')
            method_name = method_name.replace('-preview-04-17', '')
            method_name = method_name.replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Store the raw duplicate counts (equivalent to 'top_obs' in performance data)
            duplicate_data[dataset_name.lower()][path] = {
                'dups': stats['dups'],  # Raw duplicate counts for each run
                'median': stats['median'],
                'q1': stats['q1'],
                'q3': stats['q3']
            }

    # Sort by dataset order
    sorted_dataset_names = sorted(dataset_to_color.keys(), key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), reverse=True)
    duplicate_data = {k: duplicate_data[k] for k in sorted_dataset_names if k in duplicate_data}

    # Group methods by provider for boxplots
    method_data = {}
    method_names = []
    # FILTER_METHODS = False  # Set to False to include all methods
    remove_datasets = []

    # First collect all method names and their dataset coverage
    for dataset_name, dataset_data in duplicate_data.items():
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

    # Group methods by provider
    provider_methods = {}
    for provider, models_list in model_to_provider.items():
        # Only include methods that are in our filtered method_names list
        provider_filtered_methods = [method for method in models_list if method in method_names]
        if provider_filtered_methods:  # Only include providers that have methods
            provider_methods[provider] = provider_filtered_methods
        else:
            print(f'No methods found for provider: {provider}')

    # Create individual plots for each provider (for duplicates)
    print("Creating duplicate suggestion plots for each provider...")
    remove_datasets = []

    all_provider_duplicate_data = {}
    for provider_name, methods in provider_methods.items():
        print(f"\nCreating duplicate plot for {provider_name}...")
        
        # Filter methods that have duplicate data
        available_methods = []
        for method in methods:
            method_found = False
            for dataset_name, dataset_data in duplicate_data.items():
                for key in dataset_data.keys():
                    current_method = key.split('/')[-1]
                    current_method = current_method.replace('-1-20-20', '')
                    current_method = current_method.replace('-latest', '')
                    current_method = current_method.replace('-preview-03-25', '')
                    current_method = current_method.replace('-20250514', '')
                    current_method = current_method.replace('-preview-04-17', '')
                    current_method = current_method.replace('-des0', '-des')
                    current_method = current_method.replace('-preview-06-17', '')
                    if current_method == method:
                        method_found = True
                        break
                if method_found:
                    break
            if method_found:
                available_methods.append(method)
        
        if available_methods:
            fig, ax, provider_duplicate_data = create_individual_provider_duplicate_plot(
                provider_name, available_methods, remove_datasets
            )
            all_provider_duplicate_data[provider_name] = provider_duplicate_data
            # Optional: Save the figure
            os.makedirs('./pngs', exist_ok=True)
            plt.savefig(f'./pngs/figure_6_{provider_name}.png', dpi=300, bbox_inches='tight')
        else:
            print(f"No duplicate data found for {provider_name}")

    print(f"\nCreated duplicate suggestion plots for {len(all_provider_duplicate_data)} providers.")

    # Print summary
    print(f"\nSummary of plotted providers (duplicates):")
    for provider, methods in provider_methods.items():
        if provider in all_provider_duplicate_data:
            provider_data = all_provider_duplicate_data[provider]
            d1 =list(provider_data.keys())[0]
            available_methods = len(provider_data[d1])
            print(f"  {provider}: {available_methods} methods with duplicate data")

    print(f'Figure 6 Anthropic saved to ./pngs/figure_6_Anthropic.png')
    print(f'Figure 6 Google saved to ./pngs/figure_6_Google.png')
    print(f'Figure 6 OpenAI saved to ./pngs/figure_6_OpenAI.png')
    print(f'Figure 6 Atlas saved to ./pngs/figure_6_Atlas.png')
