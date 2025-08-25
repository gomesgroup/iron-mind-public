import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.stats import spearmanr
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

def get_objective_value(obs_or_group, dataset_config, aggregation='first'):
    """
    Extract objective value from observation(s) based on dataset configuration
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
            objectives = []
            for obj_name in dataset_config['objectives']:
                if aggregation == 'first':
                    objectives.append(obs_or_group[obj_name].iloc[0])
                elif aggregation == 'mean':
                    objectives.append(obs_or_group[obj_name].mean())
                elif aggregation == 'max':
                    objectives.append(obs_or_group[obj_name].max())
                elif aggregation == 'min':
                    objectives.append(obs_or_group[obj_name].min())
                elif aggregation == 'median':
                    objectives.append(obs_or_group[obj_name].median())
        else:
            # Single observation
            objectives = [obs_or_group[obj_name] for obj_name in dataset_config['objectives']]
        
        # Apply transformation
        if dataset_config['transform'] == 'subtract':
            order = dataset_config['order']
            result = objectives[order[0]] - objectives[order[1]]
        else:
            raise ValueError(f"Unknown transformation: {dataset_config['transform']}")
        
        return result

def compute_individual_run_cumulative_entropy(run_selections, param_options):
    """
    Compute cumulative entropy for a single run across all iterations.
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

def get_performance_and_best_iteration(run_dir, dataset_name, is_bo=False):
    """Get the best performance and the iteration where it was achieved"""
    seen_path = os.path.join(run_dir, 'seen_observations.json')
    
    if not os.path.exists(seen_path):
        return None, None
    
    with open(seen_path, 'r') as f:
        seen_data = json.load(f)
    
    dataset_config = dataset_to_obj[dataset_name]
    objective_values = []
    
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
        
        # Process groups and extract objectives
        for name, group in param_groups:
            try:
                obj_value = get_objective_value(group, dataset_config, 'mean')
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
                obj_value = get_objective_value(obs, dataset_config)
                if np.isnan(obj_value):
                    objective_values.append(0)
                else:
                    objective_values.append(obj_value)
            except (KeyError, TypeError, ZeroDivisionError):
                objective_values.append(0)
    
    # Convert to array and handle BO case
    objective_array = np.array(objective_values)
    if is_bo:
        # Skip the first observation
        objective_array = objective_array[1:]
    
    # Find best performance and iteration
    if len(objective_array) > 0:
        best_performance = float(np.max(objective_array))
        best_iteration = int(np.argmax(objective_array)) + 1  # Convert to 1-based indexing
        return best_performance, best_iteration
    else:
        return None, None

def compute_entropy_to_best(run_selections, param_options, best_iteration):
    """
    Compute cumulative entropy only up to the iteration where best performance was achieved
    """
    if not run_selections or best_iteration <= 0:
        return 0.0
    
    # Only use selections up to best_iteration
    relevant_selections = run_selections[:best_iteration]
    
    if not relevant_selections:
        return 0.0
    
    return compute_individual_run_cumulative_entropy(relevant_selections, param_options)

def extract_entropy_performance_data(path_dict, dataset_param_options):
    """Extract entropy and performance data for each individual run"""
    correlation_data = {}
    
    for dataset_name, optimization_data in path_dict.items():
        correlation_data[dataset_name.lower()] = {}
        param_options = dataset_param_options[dataset_name]
        objectives = dataset_to_obj[dataset_name]
        
        if isinstance(objectives, dict):
            objectives = objectives["objectives"]
        else:
            objectives = [objectives]
        
        for method_path, _ in optimization_data.items():
            # Determine if this is a BO method
            is_bo_method = 'bayesian' in method_path
            
            # Get all run directories for this method
            runs = glob.glob(method_path + "/run_*/")
            
            entropies = []
            performances = []
            
            for run_dir in runs:
                # Extract parameter selections for entropy calculation
                seen_path = os.path.join(run_dir, 'seen_observations.json')
                if not os.path.exists(seen_path):
                    continue
                
                with open(seen_path, 'r') as f:
                    seen_data = json.load(f)
                
                # Extract parameter selections (exclude objective columns)
                param_selections = []
                for obs in seen_data:
                    selection = {k: v for k, v in obs.items() 
                               if k in param_options.keys()}
                    param_selections.append(selection)
                
                # Skip first selection for BO methods
                if is_bo_method and len(param_selections) > 1:
                    param_selections = param_selections[1:]
                
                # Get best performance and iteration where it occurred
                performance, best_iteration = get_performance_and_best_iteration(run_dir, dataset_name, is_bo_method)
                
                if performance is not None and best_iteration is not None:
                    # Calculate entropy only up to the iteration where best performance was achieved
                    entropy = compute_entropy_to_best(param_selections, param_options, best_iteration)
                    
                    if entropy is not None:
                        entropies.append(entropy)
                        performances.append(performance)
            
            if entropies and performances:
                correlation_data[dataset_name.lower()][method_path] = {
                    'entropies': entropies,
                    'performances': performances,
                    'n_runs': len(entropies)
                }
    
    return correlation_data

def create_entropy_performance_scatter_plots(correlation_data, save_path="./pngs/"):
    """Create 6x4 grid scatter plots for each dataset showing entropy vs performance per method"""
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get all unique methods across datasets
    all_methods = set()
    for dataset_data in correlation_data.values():
        for method_key in dataset_data.keys():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
            all_methods.add(method_name)
    
    # Create a consistent ordering of methods (by provider, then alphabetically)
    ordered_methods = []
    for provider, models_list in model_to_provider.items():
        provider_methods = [method for method in models_list if method in all_methods]
        ordered_methods.extend(sorted(provider_methods))
    
    # Sort datasets by color order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Dataset name mapping for clean display
    dataset_name_mapping = {
        'suzuki_doyle': 'Suzuki Yield',
        'suzuki_cernak': 'Suzuki Conversion', 
        'chan_lam_full': 'Chan-Lam',
        'buchwald_hartwig': 'Buchwald-Hartwig',
        'reductive_amination': 'Reductive Amination',
        'alkylation_deprotection': 'Alkylation Deprotection'
    }
    
    # Create one figure per dataset
    for dataset_key in sorted_dataset_names:
        if dataset_key not in correlation_data:
            continue
            
        dataset_data = correlation_data[dataset_key]
        display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
        
        print(f"\nCreating entropy-performance correlation plot for {display_name}...")
        
        # Create 6x4 grid (24 methods)
        fig, axes = plt.subplots(4, 6, figsize=(30, 20))  # Wide figure for 6 columns
        axes = axes.flatten()
        
        method_count = 0
        for method_idx, method_name in enumerate(ordered_methods):
            if method_count >= 24:  # Limit to 24 subplots
                break
                
            ax = axes[method_count]
            
            # Find matching method in the dataset data
            method_data = None
            for method_key, data in dataset_data.items():
                key_method = method_key.split('/')[-1]
                key_method = key_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                key_method = key_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                key_method = key_method.replace('-preview-06-17', '')
                
                if key_method == method_name:
                    method_data = data
                    break
            
            if method_data is not None:
                entropies = method_data['entropies']
                performances = method_data['performances']
                n_runs = len(entropies)
                
                # Determine method color based on provider
                method_color = '#1f77b4'  # Default blue
                for provider, models_list in model_to_provider.items():
                    if method_name in models_list:
                        if provider == 'Anthropic':
                            method_color = '#ff7f0e'  # Orange
                        elif provider == 'Google':
                            method_color = '#2ca02c'  # Green
                        elif provider == 'OpenAI':
                            method_color = '#d62728'  # Red
                        elif provider == 'Atlas':
                            method_color = '#9467bd'  # Purple
                        break
                
                # Create scatter plot (should be ~20 points per method)
                ax.scatter(entropies, performances, c=method_color, alpha=0.7, s=50)
                
                # Calculate and display Spearman correlation
                if len(entropies) > 1:
                    correlation, p_value = spearmanr(entropies, performances)
                    
                    # Add correlation text at bottom of subplot
                    correlation_text = f'ρ={correlation:.2f}'
                    if p_value < 0.001:
                        correlation_text += '***'
                    elif p_value < 0.01:
                        correlation_text += '**'
                    elif p_value < 0.05:
                        correlation_text += '*'
                    
                    ax.text(0.5, 0.02, correlation_text, transform=ax.transAxes, 
                           fontsize=10, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Add sample size annotation
                ax.text(0.98, 0.98, f'n={n_runs}', transform=ax.transAxes, 
                       fontsize=9, ha='right', va='top', alpha=0.7)
                
            else:
                # No data for this method on this dataset
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                       fontsize=12, ha='center', va='center', alpha=0.5)
            
            # Customize subplot
            ax.set_title(f'{method_name}', fontsize=11, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            
            # Only add axis labels to bottom and left edges
            if method_count >= 18:  # Bottom row
                ax.set_xlabel('Entropy-to-Best', fontsize=10)
            if method_count % 6 == 0:  # Left column
                ax.set_ylabel('Best Performance (%)', fontsize=10)
            
            method_count += 1
        
        # Hide unused subplots
        for idx in range(method_count, 24):
            axes[idx].set_visible(False)
        
        # Add overall title
        plt.suptitle(f'{display_name} - Method-Specific Entropy vs Performance Correlation', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.4, wspace=0.3)
        
        # Save the plot
        filename = f'figure_S9_{dataset_key}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {filename}")

def create_correlation_summary_table(correlation_data):
    """Create a summary table of correlations for manuscript"""
    
    # Group methods by provider
    provider_methods = {}
    all_methods = set()
    
    for dataset_data in correlation_data.values():
        for method_key in dataset_data.keys():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
            all_methods.add(method_name)
    
    # Group by provider
    for provider, models_list in model_to_provider.items():
        provider_filtered_methods = [method for method in models_list if method in all_methods]
        if provider_filtered_methods:
            provider_methods[provider] = provider_filtered_methods
    
    print("\n" + "="*80)
    print("ENTROPY-PERFORMANCE CORRELATION SUMMARY")
    print("="*80)
    
    # Calculate provider-level correlations
    provider_correlations = {}
    
    for provider_name, methods in provider_methods.items():
        all_entropies = []
        all_performances = []
        
        # Collect all data for this provider
        for method_name in methods:
            for dataset_key, dataset_data in correlation_data.items():
                for method_key, data in dataset_data.items():
                    key_method = method_key.split('/')[-1]
                    key_method = key_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                    key_method = key_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                    key_method = key_method.replace('-preview-06-17', '')
                    
                    if key_method == method_name:
                        all_entropies.extend(data['entropies'])
                        all_performances.extend(data['performances'])
        
        if len(all_entropies) > 1:
            correlation, p_value = spearmanr(all_entropies, all_performances)
            provider_correlations[provider_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'n_points': len(all_entropies)
            }
    
    # Display results
    print(f"\n{'Provider':<12} {'ρ (Spearman)':<15} {'p-value':<10} {'n':<8} {'Interpretation'}")
    print("-" * 70)
    
    for provider, stats in provider_correlations.items():
        correlation = stats['correlation']
        p_value = stats['p_value']
        n_points = stats['n_points']
        
        # Interpret correlation strength
        if abs(correlation) < 0.1:
            interpretation = "negligible"
        elif abs(correlation) < 0.3:
            interpretation = "weak"
        elif abs(correlation) < 0.5:
            interpretation = "moderate"
        elif abs(correlation) < 0.7:
            interpretation = "strong"
        else:
            interpretation = "very strong"
        
        # Add significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"{provider:<12} {correlation:>7.3f} {sig:<7} {p_value:>7.3f} {n_points:>6} {interpretation}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    # Interpretation for decoupling claim
    print("\n" + "="*80)
    print("DECOUPLING ANALYSIS INTERPRETATION:")
    print("="*80)
    
    llm_correlations = []
    bo_correlations = []
    
    for provider, stats in provider_correlations.items():
        if provider in ['Anthropic', 'Google', 'OpenAI']:
            llm_correlations.append(stats['correlation'])
        elif provider == 'Atlas':
            bo_correlations.append(stats['correlation'])
    
    if llm_correlations:
        avg_llm_corr = np.mean(llm_correlations)
        print(f"Average LLM correlation: ρ = {avg_llm_corr:.3f}")
    
    if bo_correlations:
        avg_bo_corr = np.mean(bo_correlations)
        print(f"Average BO correlation: ρ = {avg_bo_corr:.3f}")
    
    print("\nDECOUPLING CLAIM ASSESSMENT:")
    if llm_correlations and bo_correlations:
        if abs(avg_llm_corr) < abs(avg_bo_corr):
            print("✓ SUPPORTS decoupling claim: LLM methods show weaker entropy-performance correlation")
            print("  → Different exploration strategies can achieve similar performance")
        else:
            print("✗ CONTRADICTS decoupling claim: LLM methods show similar/stronger correlation than BO")
            print("  → Strategy and performance appear coupled like in BO methods")
    
    return provider_correlations

if __name__ == "__main__":
    # Get runs path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the runs path: ')
    
    # Load dataset parameter options (same as figure_7_new.py)
    dataset_param_options = {}
    for dataset_name in dataset_names:
        param_options = {}
        dataset = Dataset(dataset_name)
        for param in dataset.param_space:
            param_options[param.name] = param.options
        dataset_param_options[dataset_name] = param_options
    
    # Build path dictionary
    path_dict = {}
    for dataset_name in dataset_names:
        path_dict[dataset_name] = {}
        
        for method in ['bayesian', 'llm']:
            method_path = os.path.join(run_path, f'{method}/{dataset_name}/benchmark/')
            if os.path.exists(method_path):
                paths = [os.path.join(method_path, sub_path) for sub_path in os.listdir(method_path)]
                paths = [path for path in paths if path.endswith('1-20-20') or path.endswith('1-20-20-des0')]
                
                for path in paths:
                    path_dict[dataset_name][path] = None  # We don't need tracks for this analysis
    
    print("Extracting entropy-performance correlation data...")
    correlation_data = extract_entropy_performance_data(path_dict, dataset_param_options)
    
    print("Creating scatter plots...")
    create_entropy_performance_scatter_plots(correlation_data)
    
    print(f"\nAnalysis complete! Check ./pngs/ for scatter plot figures.")

    print(f'Figure S9 Buchwald_Hartwig saved to ./pngs/figure_S9_Buchwald_Hartwig.png')
    print(f'Figure S9 Suzuki_Doyle saved to ./pngs/figure_S9_Suzuki_Doyle.png')
    print(f'Figure S9 Suzuki_Cernak saved to ./pngs/figure_S9_Suzuki_Cernak.png')
    print(f'Figure S9 Reductive_Amination saved to ./pngs/figure_S9_Reductive_Amination.png')
    print(f'Figure S9 Alkylation_Deprotection saved to ./pngs/figure_S9_Alkylation_Deprotection.png')
    print(f'Figure S9 Chan_Lam_Full saved to ./pngs/figure_S9_Chan_Lam_Full.png')
