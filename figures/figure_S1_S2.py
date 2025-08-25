import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Import the statistical functions and data loading from figure_5
from figure_5_S12 import (
    get_top_obs_data, get_tracks, dataset_names, dataset_to_obj, model_to_provider,
    dataset_to_color, dataset_order, perform_statistical_analysis, get_provider_data,
    find_best_llm_vs_best_bo_per_dataset
)

def create_individual_statistical_matrices(top_obs_data, save_path="./pngs/individual/"):
    """Create individual statistical comparison matrices for each dataset"""
    
    # Apply same method filtering as figure_5.py
    filtered_top_obs_data = {}
    for dataset_name, dataset_data in top_obs_data.items():
        filtered_top_obs_data[dataset_name] = {}
        for key, method_info in dataset_data.items():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
                
            filtered_top_obs_data[dataset_name][key] = method_info
    
    # Sort datasets by color order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Set font
    plt.rcParams['font.family'] = 'SF Pro Display'
    
    # Dataset name mapping for clean display
    dataset_name_mapping = {
        'suzuki_doyle': 'Suzuki Yield',
        'suzuki_cernak': 'Suzuki Conversion', 
        'chan_lam_full': 'Chan-Lam',
        'buchwald_hartwig': 'Buchwald-Hartwig',
        'reductive_amination': 'Reductive Amination',
        'alkylation_deprotection': 'Alkylation Deprotection'
    }
    
    # Create matrices for each dataset
    for dataset_key in sorted_dataset_names:
        if dataset_key not in filtered_top_obs_data:
            continue
            
        dataset_data = filtered_top_obs_data[dataset_key]
        
        # Separate LLM and BO methods
        llm_methods = {}
        bo_methods = {}
        
        for method_key, method_info in dataset_data.items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Classify as LLM or BO
            is_llm = any(method_name in provider_methods for provider, provider_methods in model_to_provider.items() 
                        if provider in ['Anthropic', 'Google', 'OpenAI'])
            is_bo = method_name in model_to_provider['Atlas']
            
            if is_llm:
                llm_methods[method_name] = method_info['top_obs']
            elif is_bo:
                bo_methods[method_name] = method_info['top_obs']
        
        if not llm_methods or not bo_methods:
            print(f"Skipping {dataset_key}: insufficient LLM or BO methods")
            continue
        
        # Create matrices for this dataset
        llm_names = sorted(llm_methods.keys())
        bo_names = sorted(bo_methods.keys())
        
        p_matrix = np.zeros((len(llm_names), len(bo_names)))
        effect_matrix = np.zeros((len(llm_names), len(bo_names)))
        
        for i, llm_name in enumerate(llm_names):
            for j, bo_name in enumerate(bo_names):
                llm_data = llm_methods[llm_name]
                bo_data = bo_methods[bo_name]
                
                p_val, effect_size, _ = perform_statistical_analysis(llm_data, bo_data, llm_name, bo_name)
                
                p_matrix[i, j] = p_val if p_val is not None else 1.0
                effect_matrix[i, j] = effect_size if effect_size is not None else 0.0
        
        # Create individual plots for this dataset
        create_individual_pvalue_plot(dataset_key, dataset_name_mapping, p_matrix, 
                                     llm_names, bo_names, save_path)
        create_individual_effect_plot(dataset_key, dataset_name_mapping, effect_matrix, 
                                     llm_names, bo_names, save_path)

def create_individual_pvalue_plot(dataset_key, dataset_name_mapping, p_matrix, 
                                llm_names, bo_names, save_path):
    """Create individual p-value plot for a single dataset"""
    
    # Get clean display name
    display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
    
    # Create figure with slightly more width for better cell spacing
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create custom colormap for p-values
    colors = ['#d73027', '#fee08b', '#abdda4', '#2b83ba']  # Red, Light Yellow, Light Green, Dark Blue
    p_cmap = ListedColormap(colors)
    
    p_colors = np.zeros_like(p_matrix)
    p_colors[p_matrix >= 0.05] = 0  # Red (not significant)
    p_colors[(p_matrix >= 0.01) & (p_matrix < 0.05)] = 1  # Light yellow-green
    p_colors[(p_matrix >= 0.001) & (p_matrix < 0.01)] = 2  # Medium green
    p_colors[p_matrix < 0.001] = 3  # Dark blue (highly significant)
    
    im = ax.imshow(p_colors, cmap=p_cmap, aspect='auto', vmin=0, vmax=3)
    
    # Add p-value text annotations
    for i in range(len(llm_names)):
        for j in range(len(bo_names)):
            p_val = p_matrix[i, j]
            if p_val < 0.001:
                text = f'{p_val:.2e}***'
            elif p_val < 0.01:
                text = f'{p_val:.2f}**'
            elif p_val < 0.05:
                text = f'{p_val:.2f}*'
            else:
                text = f'{p_val:.2f}'
            
            # Adaptive text color based on background
            color_value = p_colors[i, j]
            if color_value == 0:  # Red background
                text_color = 'white'
            elif color_value == 1:  # Light yellow background  
                text_color = 'black'
            elif color_value == 2:  # Medium green background
                text_color = 'black'
            else:  # Dark blue background (color_value == 3)
                text_color = 'white'
            
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color=text_color)
    
    ax.set_title(f'{display_name} - Statistical Significance (Wilcoxon Test)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(bo_names)))
    ax.set_yticks(range(len(llm_names)))
    ax.set_xticklabels(bo_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(llm_names, fontsize=12)
    ax.set_xlabel('BO Methods', fontweight='bold', fontsize=14)
    ax.set_ylabel('LLM Methods', fontweight='bold', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Significance Level', fontweight='bold', fontsize=12)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['p≥0.05 (ns)', 'p<0.05 (*)', 'p<0.01 (**)', 'p<0.001 (***)'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    safe_dataset_name = dataset_key.replace('_', '-')
    filename = f'figure_S1_{safe_dataset_name}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()

def create_individual_effect_plot(dataset_key, dataset_name_mapping, effect_matrix, 
                                llm_names, bo_names, save_path):
    """Create individual effect size plot for a single dataset"""
    
    # Get clean display name
    display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
    
    # Create figure with slightly more width for better cell spacing
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use diverging colormap for effect sizes (centered at 0)
    vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max())) if effect_matrix.size > 0 else 1
    im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', 
                  vmin=-vmax, vmax=vmax)
    
    # Add effect size text annotations
    for i in range(len(llm_names)):
        for j in range(len(bo_names)):
            effect = effect_matrix[i, j]
            text = f'{effect:.2f}'
            
            # Color text based on background
            text_color = 'white' if abs(effect) > vmax * 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=12, fontweight='bold', color=text_color)
    
    ax.set_title(f'{display_name} - Effect Sizes (Cliff\'s δ)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(bo_names)))
    ax.set_yticks(range(len(llm_names)))
    ax.set_xticklabels(bo_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(llm_names, fontsize=12)
    ax.set_xlabel('BO Methods', fontweight='bold', fontsize=14)
    ax.set_ylabel('LLM Methods', fontweight='bold', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Cliff\'s δ (LLM advantage →)', fontweight='bold', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    safe_dataset_name = dataset_key.replace('_', '-')
    filename = f'figure_S2_{safe_dataset_name}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Get run path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')
    
    # Load data using the same logic as figure_5.py
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

    # Get top observations data
    top_obs_data = get_top_obs_data(path_dict)
    
    # Sort by dataset order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    top_obs_data = {k: top_obs_data[k] for k in sorted_dataset_names}
    
    print("Creating individual statistical matrices (separate p-values and effect sizes)...")
    create_individual_statistical_matrices(top_obs_data, save_path="./pngs/")
    print("Individual p-value and effect size matrices saved to './pngs/'")
    
    print("\nAll individual figures have been generated successfully!")

    print(f'Figure S1 Buchwald_Hartwig saved to ./pngs/figure_S1_Buchwald_Hartwig.png')
    print(f'Figure S1 Suzuki_Doyle saved to ./pngs/figure_S1_Suzuki_Doyle.png')
    print(f'Figure S1 Suzuki_Cernak saved to ./pngs/figure_S1_Suzuki_Cernak.png')
    print(f'Figure S1 Reductive_Amination saved to ./pngs/figure_S1_Reductive_Amination.png')
    print(f'Figure S1 Alkylation_Deprotection saved to ./pngs/figure_S1_Alkylation_Deprotection.png')
    print(f'Figure S1 Chan_Lam_Full saved to ./pngs/figure_S1_Chan_Lam_Full.png')
    print(f'Figure S2 Buchwald_Hartwig saved to ./pngs/figure_S2_Buchwald_Hartwig.png')
    print(f'Figure S2 Suzuki_Doyle saved to ./pngs/figure_S2_Suzuki_Doyle.png')
    print(f'Figure S2 Suzuki_Cernak saved to ./pngs/figure_S2_Suzuki_Cernak.png')
