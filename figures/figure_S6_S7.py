#!/usr/bin/env python3
"""
Create individual statistical matrices for entropy data analysis,
focusing on Wilcoxon p-values for LLM vs BO comparisons.
Based on entropy_distribution_analysis.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob, os
import sys
from olympus.datasets.dataset import Dataset
from matplotlib.colors import ListedColormap

# Import functions from figure_7_new.py and figure_5.py
import importlib.util
spec = importlib.util.spec_from_file_location("figure_7", "figures/figure_7.py")
figure_7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure_7)

from figure_5_S12 import perform_statistical_analysis

def setup_data(run_path):
    """Setup path_dict and dataset_param_options similar to entropy_distribution_analysis.py"""
    
    dataset_names = [
        'Buchwald_Hartwig', 
        'Suzuki_Doyle', 
        'Suzuki_Cernak', 
        'Reductive_Amination', 
        'Alkylation_Deprotection', 
        'Chan_Lam_Full'
    ]
    
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
    
    return path_dict, dataset_param_options

def create_individual_entropy_statistical_matrices(entropy_data, save_path="./pngs/individual/"):
    """Create individual statistical comparison matrices for entropy data - p-values only"""
    
    # Define provider mapping (same as entropy_distribution_analysis.py)
    model_to_provider = {
        "Anthropic": [
            'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-3-7-sonnet',
            'claude-3-7-sonnet-thinking', 'claude-sonnet-4', 'claude-opus-4',
        ],
        "Google": [
            'gemini-2.0-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-flash',
            'gemini-2.5-flash-medium', 'gemini-2.5-pro', 'gemini-2.5-pro-medium',
        ],
        "OpenAI": [
            'gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1', 'o4-mini-low', 'o3-low',
        ],
        "Atlas": [
            'atlas-ei', 'atlas-ei-des', 'atlas-ucb', 'atlas-ucb-des', 'atlas-pi', 'atlas-pi-des',
        ]
    }
    
    # Dataset ordering (same as entropy analysis)
    dataset_to_color = {
        'reductive_amination': '#221150',
        'buchwald_hartwig': '#5e177f',
        'chan_lam_full': '#972c7f',
        'suzuki_cernak': '#d3426d',
        'suzuki_doyle': '#f8755c',
        'alkylation_deprotection': '#febb80'
    }
    
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
        if dataset_key not in entropy_data:
            continue
            
        dataset_data = entropy_data[dataset_key]
        
        # Separate LLM and BO methods
        llm_methods = {}
        bo_methods = {}
        
        for method_path, data in dataset_data.items():
            if not data['cumulative_entropies']:
                continue
                
            method_name = method_path.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Filter out nano methods
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
            
            # Classify as LLM or BO
            is_llm = any(method_name in provider_methods for provider, provider_methods in model_to_provider.items() 
                        if provider in ['Anthropic', 'Google', 'OpenAI'])
            is_bo = method_name in model_to_provider['Atlas']
            
            if is_llm:
                llm_methods[method_name] = data['cumulative_entropies']
            elif is_bo:
                bo_methods[method_name] = data['cumulative_entropies']
        
        if not llm_methods or not bo_methods:
            print(f"Skipping {dataset_key}: insufficient LLM or BO methods for entropy analysis")
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
        create_individual_entropy_pvalue_plot(dataset_key, dataset_name_mapping, p_matrix, 
                                            llm_names, bo_names, save_path)
        create_individual_entropy_effect_plot(dataset_key, dataset_name_mapping, effect_matrix, 
                                            llm_names, bo_names, save_path)

def create_individual_entropy_pvalue_plot(dataset_key, dataset_name_mapping, p_matrix, 
                                        llm_names, bo_names, save_path):
    """Create individual entropy p-value plot for a single dataset"""
    
    # Get clean display name
    display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
    
    # Create figure with slightly more width for better cell spacing
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create custom colormap for p-values (same as individual_statistical_matrices.py)
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
    
    ax.set_title(f'{display_name} - Entropy Statistical Significance (Wilcoxon Test)', 
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
    filename = f'figure_S6_{safe_dataset_name}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()

def create_individual_entropy_effect_plot(dataset_key, dataset_name_mapping, effect_matrix, 
                                        llm_names, bo_names, save_path):
    """Create individual entropy effect size plot for a single dataset"""
    
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
    
    ax.set_title(f'{display_name} - Entropy Effect Sizes (Cliff\'s δ)', 
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
    filename = f'figure_S7_{safe_dataset_name}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()
    plt.close()

def main():
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input("Enter the path to the runs directory: ")
    
    if not os.path.exists(run_path):
        print(f"Error: Path {run_path} does not exist.")
        return
    
    print("Setting up dataset paths and parameters...")
    
    # Setup data structures
    path_dict, dataset_param_options = setup_data(run_path)
    
    # Debug: Print what paths were found
    print("Debug - Paths found:")
    for dataset_name, paths in path_dict.items():
        print(f"  {dataset_name}: {len(paths)} paths")
        for path in list(paths.keys())[:3]:  # Show first 3 paths
            method_name = path.split('/')[-1]
            print(f"    {method_name}")
        if len(paths) > 3:
            print(f"    ... and {len(paths) - 3} more")
    
    print("Extracting cumulative entropy data...")
    entropy_data = figure_7.extract_cumulative_entropy_data(path_dict, dataset_param_options)
    
    print("\nCreating individual entropy statistical matrices (p-values and effect sizes)...")
    create_individual_entropy_statistical_matrices(entropy_data, save_path="./pngs/")
    print("Individual entropy statistical matrices saved to './pngs/'")
    print("Files generated:")
    print("  - entropy_pvalue_matrix_[dataset].png (entropy p-value matrices)")
    print("  - entropy_effect_matrix_[dataset].png (entropy effect size matrices)")
    
    print("\nAll individual entropy figures have been generated successfully!")

    print(f'Figure S6 Buchwald_Hartwig saved to ./pngs/figure_S6_Buchwald_Hartwig.png')
    print(f'Figure S6 Suzuki_Doyle saved to ./pngs/figure_S6_Suzuki_Doyle.png')
    print(f'Figure S6 Suzuki_Cernak saved to ./pngs/figure_S6_Suzuki_Cernak.png')
    print(f'Figure S6 Reductive_Amination saved to ./pngs/figure_S6_Reductive_Amination.png')
    print(f'Figure S6 Alkylation_Deprotection saved to ./pngs/figure_S6_Alkylation_Deprotection.png')
    print(f'Figure S6 Chan_Lam_Full saved to ./pngs/figure_S6_Chan_Lam_Full.png')
    print(f'Figure S7 Buchwald_Hartwig saved to ./pngs/figure_S7_Buchwald_Hartwig.png')
    print(f'Figure S7 Suzuki_Doyle saved to ./pngs/figure_S7_Suzuki_Doyle.png')
    print(f'Figure S7 Suzuki_Cernak saved to ./pngs/figure_S7_Suzuki_Cernak.png')
    print(f'Figure S7 Reductive_Amination saved to ./pngs/figure_S7_Reductive_Amination.png')
    print(f'Figure S7 Alkylation_Deprotection saved to ./pngs/figure_S7_Alkylation_Deprotection.png')
    print(f'Figure S7 Chan_Lam_Full saved to ./pngs/figure_S7_Chan_Lam_Full.png')

if __name__ == "__main__":
    main()
