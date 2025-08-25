#!/usr/bin/env python3
"""
Standalone script to analyze cumulative entropy median distributions
to identify methods that stand out as significantly different.
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

# Import functions from figure_7.py
import importlib.util
spec = importlib.util.spec_from_file_location("figure_7_S10", "figure_7_S10.py")
figure_7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure_7)

def setup_data(run_path):
    """Setup path_dict and dataset_param_options similar to figure_7.py"""
    
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

def bootstrap_entropy_median_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for median"""
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    np.random.seed(42)  # For reproducibility
    bootstrap_medians = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    median = np.median(data)
    ci_lower = np.percentile(bootstrap_medians, lower_percentile)
    ci_upper = np.percentile(bootstrap_medians, upper_percentile)
    
    return median, ci_lower, ci_upper

def create_entropy_bootstrap_ci_table(entropy_data):
    """Create a table of bootstrap confidence intervals for entropy medians - method level"""
    
    # Define provider mapping (same as statistical_summary_table.py)
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
    
    # Dataset ordering (same as performance analysis)
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
    
    # Collect all methods across all datasets
    all_methods = set()
    for dataset_data in entropy_data.values():
        for method_path in dataset_data.keys():
            method_name = method_path.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
                
            all_methods.add(method_name)
    
    # Sort methods by provider for better organization
    provider_methods = {}
    for provider, methods_list in model_to_provider.items():
        provider_methods[provider] = [m for m in methods_list if m in all_methods]
    
    # Create table data
    table_data = []
    
    for dataset_key in sorted_dataset_names:
        if dataset_key not in entropy_data:
            continue
            
        dataset_data = entropy_data[dataset_key]
        dataset_name = dataset_key.replace('_', ' ').title()
        
        # Group methods by provider
        for provider in ['Anthropic', 'Google', 'OpenAI', 'Atlas']:
            if provider not in provider_methods or not provider_methods[provider]:
                continue
                
            for method_name in sorted(provider_methods[provider]):
                # Find data for this method in this dataset
                method_entropy_data = None
                for method_path, data in dataset_data.items():
                    path_method = method_path.split('/')[-1]
                    path_method = path_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                    path_method = path_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                    path_method = path_method.replace('-preview-06-17', '')
                    
                    if path_method == method_name and data['cumulative_entropies']:
                        method_entropy_data = data['cumulative_entropies']
                        break
                
                if method_entropy_data is not None:
                    median, ci_lower, ci_upper = bootstrap_entropy_median_ci(method_entropy_data)
                    
                    table_data.append({
                        'Dataset': dataset_name,
                        'Provider': provider,
                        'Method': method_name,
                        'Median': f"{median:.3f}",
                        'CI_Lower': f"{ci_lower:.3f}",
                        'CI_Upper': f"{ci_upper:.3f}",
                        'CI_Range': f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                        'CI_Width': f"{ci_upper - ci_lower:.3f}"
                    })
    
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(table_data)
    
    return df

def create_entropy_bootstrap_ci_figure(entropy_data):
    """Create a 3x2 subplot figure with tables showing entropy bootstrap CIs for each dataset"""
    
    # Get the CI data
    ci_df = create_entropy_bootstrap_ci_table(entropy_data)
    
    if len(ci_df) == 0:
        print("No data available for entropy bootstrap CI figure")
        return None
    
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
    
    # Sort datasets by color order (same as performance figures)
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
    
    # Create 3x2 subplot figure (same layout as bootstrap_confidence_intervals.png)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.flatten()
    
    for idx, dataset_key in enumerate(sorted_dataset_names):
        if idx >= 6:  # Only handle first 6 datasets
            break
            
        ax = axes_flat[idx]
        ax.axis('off')  # Turn off axis for table
        
        display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
        dataset_name = dataset_key.replace('_', ' ').title()
        dataset_data = ci_df[ci_df['Dataset'] == dataset_name]
        
        if len(dataset_data) == 0:
            ax.text(0.5, 0.5, f'No entropy data for {dataset_name}', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue
        
        # Prepare table data (exactly like statistical_summary_table.py)
        table_data = []
        
        # Group by provider for better organization
        providers = ['Anthropic', 'Google', 'OpenAI', 'Atlas']
        
        for provider in providers:
            provider_data = dataset_data[dataset_data['Provider'] == provider]
            if len(provider_data) == 0:
                continue
                
            # Add provider header row
            table_data.append([f'{provider}', '', '', ''])
            
            # Add methods for this provider
            for _, row in provider_data.iterrows():
                method_name = row['Method']
                
                table_data.append([
                    f'  {method_name}',  # Indent method names
                    f"{row['Median']}",  # No % for entropy
                    f"{row['CI_Lower']}",
                    f"{row['CI_Upper']}"
                ])
        
        # Create table (exactly like statistical_summary_table.py)
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=['Method', 'Median', 'CI Lower', 'CI Upper'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style the table (exactly like statistical_summary_table.py)
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Adjust column widths - make method column wider, CI columns narrower
            cellDict = table.get_celld()
            for i in range(len(table_data) + 1):  # +1 for header row
                # Method column (wider)
                cellDict[(i, 0)].set_width(0.5)
                # Median column 
                cellDict[(i, 1)].set_width(0.18)
                # CI Lower column (narrower)
                cellDict[(i, 2)].set_width(0.16)
                # CI Upper column (narrower)  
                cellDict[(i, 3)].set_width(0.16)
            
            # Apply systematic color coding to entropy values (0-1 range)
            def get_color_for_entropy(value_str):
                """Convert entropy string to color - blue (0) to red (1)"""
                try:
                    # Extract numeric value from string
                    value = float(value_str)
                    # Clamp value between 0 and 1
                    value = max(0, min(1, value))
                    # Create color gradient: blue (0) to red (1)
                    # Blue component decreases as value increases
                    blue = (1 - value)
                    # Red component increases as value increases  
                    red = value
                    # Keep green low for better contrast
                    green = 0.1
                    return (red, green, blue)
                except:
                    return (0, 0, 0)  # Black for invalid values
            
            # Apply colors to data rows (skip header row and provider rows)
            row_idx = 1
            for provider in providers:
                provider_data_subset = dataset_data[dataset_data['Provider'] == provider]
                if len(provider_data_subset) == 0:
                    continue
                
                # Skip provider header row
                row_idx += 1
                
                # Color the method rows for this provider
                for _, row in provider_data_subset.iterrows():
                    # Color median value (column 1)
                    median_color = get_color_for_entropy(row['Median'])
                    cellDict[(row_idx, 1)].set_text_props(color=median_color)
                    
                    # Color CI Lower value (column 2)
                    ci_lower_color = get_color_for_entropy(row['CI_Lower'])
                    cellDict[(row_idx, 2)].set_text_props(color=ci_lower_color)
                    
                    # Color CI Upper value (column 3)
                    ci_upper_color = get_color_for_entropy(row['CI_Upper'])
                    cellDict[(row_idx, 3)].set_text_props(color=ci_upper_color)
                    
                    row_idx += 1
            
            # Style header row
            for i in range(4):
                table[(0, i)].set_facecolor('#E6E6E6')
                table[(0, i)].set_text_props(weight='bold')
            
            # Style provider rows (bold, different background)
            row_idx = 1
            for provider in providers:
                provider_data = dataset_data[dataset_data['Provider'] == provider]
                if len(provider_data) == 0:
                    continue
                
                # Provider header row
                for i in range(4):
                    table[(row_idx, i)].set_facecolor('#D0D0D0')
                    table[(row_idx, i)].set_text_props(weight='bold')
                
                row_idx += 1 + len(provider_data)  # Skip provider methods
            
            # Add dataset title using clean display name
            ax.text(0.5, 1.02, display_name, ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Hide any unused subplots
    for idx in range(len(sorted_dataset_names), 6):
        axes_flat[idx].set_visible(False)
    
    # plt.suptitle('Bootstrap Confidence Intervals for Cumulative Entropy by Method\n(95% CI, 1000 bootstrap samples)', 
    #              fontsize=16, y=0.95)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.88)
    
    # Save figure
    os.makedirs('./pngs', exist_ok=True)
    plt.savefig('./pngs/figure_S8.png', dpi=300, bbox_inches='tight')
    print("Saved entropy bootstrap CI figure to ./pngs/figure_S8.png")
    
    return fig

def create_entropy_statistical_matrices(entropy_data):
    """Create statistical comparison matrices for LLM vs BO methods using entropy data"""
    
    # Import statistical functions from figure_5.py
    from figure_5 import perform_statistical_analysis
    
    # Define provider mapping
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
    
    # Dataset ordering (same as performance analysis)
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
    
    # Create matrices for each dataset
    p_value_matrices = {}
    effect_size_matrices = {}
    
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
        
        p_value_matrices[dataset_key] = {
            'matrix': p_matrix,
            'llm_names': llm_names,
            'bo_names': bo_names
        }
        
        effect_size_matrices[dataset_key] = {
            'matrix': effect_matrix,
            'llm_names': llm_names,
            'bo_names': bo_names
        }
    
    # Create the plots
    create_entropy_matrix_plots(p_value_matrices, effect_size_matrices, sorted_dataset_names)
    
    return p_value_matrices, effect_size_matrices

def create_entropy_matrix_plots(p_value_matrices, effect_size_matrices, sorted_dataset_names):
    """Create 3x2 subplot matrices for entropy statistical comparisons"""
    
    # Dataset name mapping for clean display
    dataset_name_mapping = {
        'suzuki_doyle': 'Suzuki Yield',
        'suzuki_cernak': 'Suzuki Conversion', 
        'chan_lam_full': 'Chan-Lam',
        'buchwald_hartwig': 'Buchwald-Hartwig',
        'reductive_amination': 'Reductive Amination',
        'alkylation_deprotection': 'Alkylation Deprotection'
    }
    
    # Create two figures: one for p-values, one for effect sizes
    
    # Figure 1: Wilcoxon p-values
    fig1, axes1 = plt.subplots(2, 3, figsize=(28, 12))
    
    # Figure 2: Cliff's delta effect sizes
    fig2, axes2 = plt.subplots(2, 3, figsize=(28, 12))
    
    # Flatten axes for easier indexing
    axes1_flat = axes1.flatten()
    axes2_flat = axes2.flatten()
    
    for idx, dataset_key in enumerate(sorted_dataset_names):
        if idx >= 6:  # Only handle first 6 datasets
            break
            
        if dataset_key not in p_value_matrices:
            # Hide empty subplots
            axes1_flat[idx].set_visible(False)
            axes2_flat[idx].set_visible(False)
            continue
        
        display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
        
        # Get data
        p_data = p_value_matrices[dataset_key]
        effect_data = effect_size_matrices[dataset_key]
        
        p_matrix = p_data['matrix']
        effect_matrix = effect_data['matrix']
        llm_names = p_data['llm_names']
        bo_names = p_data['bo_names']
        
        # Plot 1: P-values with significance coloring
        ax1 = axes1_flat[idx]
        
        # Create custom colormap for p-values (same as statistical_summary_table.py)
        from matplotlib.colors import ListedColormap
        
        # Define custom colors: Red (not sig) -> Light Green -> Medium Green -> Dark Green (highly sig)
        colors = ['#d73027', '#fee08b', '#abdda4', '#2b83ba']  # Red, Light Yellow, Light Green, Dark Blue
        p_cmap = ListedColormap(colors)
        
        p_colors = np.zeros_like(p_matrix)
        p_colors[p_matrix >= 0.05] = 0  # Red (not significant)
        p_colors[(p_matrix >= 0.01) & (p_matrix < 0.05)] = 1  # Light yellow-green
        p_colors[(p_matrix >= 0.001) & (p_matrix < 0.01)] = 2  # Medium green
        p_colors[p_matrix < 0.001] = 3  # Dark blue (highly significant)
        
        im1 = ax1.imshow(p_colors, cmap=p_cmap, aspect='auto', vmin=0, vmax=3)
        
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
                
                ax1.text(j, i, text, ha='center', va='center', 
                        fontsize=8, fontweight='bold', color=text_color)
        
        ax1.set_title(display_name, fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(bo_names)))
        ax1.set_yticks(range(len(llm_names)))
        ax1.set_xticklabels(bo_names, rotation=0, ha='center')
        ax1.set_yticklabels(llm_names)
        ax1.set_xlabel('BO Methods', fontweight='bold')
        ax1.set_ylabel('LLM Methods', fontweight='bold')
        
        # Plot 2: Effect sizes
        ax2 = axes2_flat[idx]
        
        # Use diverging colormap for effect sizes (centered at 0)
        vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max())) if effect_matrix.size > 0 else 1
        im2 = ax2.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', 
                        vmin=-vmax, vmax=vmax)
        
        # Add effect size text annotations
        for i in range(len(llm_names)):
            for j in range(len(bo_names)):
                effect = effect_matrix[i, j]
                text = f'{effect:.2f}'
                
                # Color text based on background
                text_color = 'white' if abs(effect) > vmax * 0.5 else 'black'
                ax2.text(j, i, text, ha='center', va='center', 
                        fontsize=9, fontweight='bold', color=text_color)
        
        ax2.set_title(display_name, fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(bo_names)))
        ax2.set_yticks(range(len(llm_names)))
        ax2.set_xticklabels(bo_names, rotation=0, ha='center')
        ax2.set_yticklabels(llm_names)
        ax2.set_xlabel('BO Methods', fontweight='bold')
        ax2.set_ylabel('LLM Methods', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(sorted_dataset_names), 6):
        axes1_flat[idx].set_visible(False)
        axes2_flat[idx].set_visible(False)
    
    # Adjust layout first to make room for colorbar
    fig1.tight_layout()
    fig2.tight_layout()
    
    # Add colorbars on the right side with proper spacing
    # P-value colorbar
    cbar1 = fig1.colorbar(im1, ax=axes1_flat, orientation='vertical', 
                         fraction=0.015, pad=0.08, shrink=0.6)
    cbar1.set_label('Significance Level', fontweight='bold')
    cbar1.set_ticks([0, 1, 2, 3])
    cbar1.set_ticklabels(['p≥0.05 (ns)', 'p<0.05 (*)', 'p<0.01 (**)', 'p<0.001 (***)'])
    
    # Effect size colorbar
    cbar2 = fig2.colorbar(im2, ax=axes2_flat, orientation='vertical', 
                         fraction=0.015, pad=0.08, shrink=0.6)
    cbar2.set_label('Cliff\'s δ (LLM advantage →)', fontweight='bold')
    
    # Final layout adjustment
    fig1.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on right for colorbar
    fig2.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on right for colorbar
    
    # Save figures
    os.makedirs('./pngs', exist_ok=True)
    fig1.savefig('./pngs/entropy_statistical_matrices_pvalues.png', 
                dpi=300, bbox_inches='tight')
    fig2.savefig('./pngs/entropy_statistical_matrices_effect_sizes.png', 
                dpi=300, bbox_inches='tight')
    
    print("Saved entropy statistical matrices:")
    print("  - './pngs/entropy_statistical_matrices_pvalues.png'")
    print("  - './pngs/entropy_statistical_matrices_effect_sizes.png'")
    
    plt.show()
    
    return fig1, fig2

def analyze_entropy_median_distribution_simple(entropy_data):
    """
    Simplified version of the entropy median distribution analysis
    that groups by provider and includes proper method filtering.
    """
    print("\n" + "="*60)
    print("CUMULATIVE ENTROPY MEDIAN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Define provider mapping (same as figure_5.py)
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
    
    # Collect all median values across datasets and methods
    all_medians = []
    method_medians = defaultdict(list)  # method -> list of medians across datasets
    dataset_medians = defaultdict(list)  # dataset -> list of medians across methods
    provider_medians = defaultdict(list)  # provider -> list of all medians from their methods
    
    for dataset_name, methods_data in entropy_data.items():
        dataset_name_display = dataset_name.replace('_', ' ').title()
        print(f"\n📊 {dataset_name_display}:")
        
        dataset_values = []
        provider_dataset_values = defaultdict(list)  # provider -> list of (method, median) for this dataset
        
        for method_path, data in methods_data.items():
            if data['median'] is not None:
                median_val = data['median']
                # Simple method name cleaning
                method_name = method_path.split('/')[-1]
                method_name = method_name.replace('-1-20-20', '')
                method_name = method_name.replace('-latest', '')
                method_name = method_name.replace('-preview-03-25', '')
                method_name = method_name.replace('-20250514', '')
                method_name = method_name.replace('-preview-04-17', '')
                method_name = method_name.replace('-des0', '-des')
                method_name = method_name.replace('-preview-06-17', '')
                
                # Filter out nano methods
                if 'gpt-4.1' in method_name and 'nano' in method_name:
                    continue
                
                # Find which provider this method belongs to
                provider_name = None
                for provider, methods_list in model_to_provider.items():
                    if method_name in methods_list:
                        provider_name = provider
                        break
                
                if provider_name:
                    all_medians.append(median_val)
                    method_medians[method_name].append(median_val)
                    provider_medians[provider_name].append(median_val)
                    dataset_values.append((method_name, median_val, provider_name))
                    provider_dataset_values[provider_name].append((method_name, median_val))
                else:
                    # Debug: Print methods that don't match any provider
                    if dataset_name == list(entropy_data.keys())[0]:  # Only print for first dataset to avoid spam
                        print(f"   ⚠️  Method '{method_name}' not found in any provider mapping")
        
        # Sort and show top/bottom methods for this dataset
        dataset_values.sort(key=lambda x: x[1], reverse=True)
        dataset_medians[dataset_name] = [val for _, val, _ in dataset_values]
        
        if dataset_values:
            print(f"   🔝 Highest Entropy: {dataset_values[0][0]} ({dataset_values[0][2]}) - {dataset_values[0][1]:.3f}")
            print(f"   🔻 Lowest Entropy:  {dataset_values[-1][0]} ({dataset_values[-1][2]}) - {dataset_values[-1][1]:.3f}")
            print(f"   📈 Range: {dataset_values[0][1] - dataset_values[-1][1]:.3f}")
            print(f"   📊 Dataset Median: {np.median([val for _, val, _ in dataset_values]):.3f}")
            
            # Show provider-level summary for this dataset
            print(f"   🏢 Provider Summary:")
            for provider in ["Anthropic", "Google", "OpenAI", "Atlas"]:
                if provider in provider_dataset_values:
                    provider_vals = [val for _, val in provider_dataset_values[provider]]
                    if provider_vals:
                        print(f"      {provider}: {np.median(provider_vals):.3f} (median of {len(provider_vals)} methods)")
    
    # Overall statistics
    print(f"\n🌍 OVERALL STATISTICS:")
    print(f"   Total median values: {len(all_medians)}")
    if all_medians:
        print(f"   Global median: {np.median(all_medians):.3f}")
        print(f"   Global mean: {np.mean(all_medians):.3f}")
        print(f"   Global std: {np.std(all_medians):.3f}")
        print(f"   Global range: {np.max(all_medians) - np.min(all_medians):.3f}")
    
    # Provider-level analysis (average across all their methods and datasets)
    print(f"\n🔬 PROVIDER-LEVEL ANALYSIS (All Methods, All Datasets):")
    provider_avg_medians = []
    for provider_name, medians in provider_medians.items():
        if len(medians) > 0:
            avg_median = np.mean(medians)
            std_median = np.std(medians) if len(medians) > 1 else 0
            provider_avg_medians.append((provider_name, avg_median, std_median, len(medians)))
    
    # Sort by average median
    provider_avg_medians.sort(key=lambda x: x[1], reverse=True)
    
    print("   Rank | Provider | Avg Median | Std | Total Values")
    print("   -----|----------|-----------|-----|-------------")
    for i, (provider, avg_med, std_med, n_values) in enumerate(provider_avg_medians):
        print(f"   {i+1:4d} | {provider:8s} | {avg_med:8.3f} | {std_med:.3f} | {n_values}")
    
    # Method-level analysis (average across datasets) - but grouped by provider
    print(f"\n🔬 METHOD-LEVEL ANALYSIS (Average Across Datasets, Grouped by Provider):")
    method_avg_medians = []
    for method_name, medians in method_medians.items():
        if len(medians) > 0:
            avg_median = np.mean(medians)
            std_median = np.std(medians) if len(medians) > 1 else 0
            # Find provider for this method
            method_provider = None
            for provider, methods_list in model_to_provider.items():
                if method_name in methods_list:
                    method_provider = provider
                    break
            method_avg_medians.append((method_name, avg_median, std_median, len(medians), method_provider))
    
    # Sort by provider, then by average median
    method_avg_medians.sort(key=lambda x: (x[4] or "ZZZ", -x[1]))
    
    current_provider = None
    for method, avg_med, std_med, n_datasets, provider in method_avg_medians:
        if provider != current_provider:
            print(f"\n   {provider}:")
            current_provider = provider
        print(f"      {method:20s}: {avg_med:6.3f} ± {std_med:.3f} ({n_datasets} datasets)")
    
    # Identify outliers using z-score (provider-level)
    if all_medians:
        global_mean = np.mean(all_medians)
        global_std = np.std(all_medians)
        threshold = 1.5  # 1.5 standard deviations (more sensitive for provider-level)
        
        print(f"\n🚨 OUTLIER PROVIDERS (|z-score| > {threshold}):")
        outlier_found = False
        for provider, avg_med, std_med, n_values in provider_avg_medians:
            z_score = (avg_med - global_mean) / global_std if global_std > 0 else 0
            if abs(z_score) > threshold:
                outlier_found = True
                direction = "HIGH" if z_score > 0 else "LOW"
                print(f"   🎯 {provider}: {avg_med:.3f} (z={z_score:+.2f}, {direction} entropy)")
        
        if not outlier_found:
            print("   No significant provider outliers found.")
    
    # Dataset-level variance analysis
    print(f"\n📊 DATASET-LEVEL VARIANCE:")
    for dataset_name, values in dataset_medians.items():
        if values:
            dataset_std = np.std(values)
            dataset_range = np.max(values) - np.min(values)
            print(f"   {dataset_name.replace('_', ' ').title():20s}: std={dataset_std:.3f}, range={dataset_range:.3f}")
    
    return {
        'all_medians': all_medians,
        'method_medians': method_medians,
        'dataset_medians': dataset_medians,
        'provider_medians': provider_medians,
        'method_rankings': method_avg_medians,
        'provider_rankings': provider_avg_medians,
        'entropy_data': entropy_data,  # Pass through the raw data for bootstrap analysis
        'global_stats': {
            'median': np.median(all_medians) if all_medians else 0,
            'mean': np.mean(all_medians) if all_medians else 0,
            'std': np.std(all_medians) if all_medians else 0
        }
    }

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
    
    print("\n" + "="*80)
    print("ANALYZING CUMULATIVE ENTROPY MEDIAN DISTRIBUTIONS...")
    print("="*80)
    
    # Run the analysis (but we need to handle the method_name_mapping issue)
    try:
        entropy_analysis = figure_7.analyze_entropy_median_distribution(entropy_data)
    except NameError as e:
        if "method_name_mapping" in str(e):
            print("⚠️  Method name mapping issue detected. Using simplified analysis...")
            entropy_analysis = analyze_entropy_median_distribution_simple(entropy_data)
        else:
            raise e
    
    # Create bootstrap confidence interval analysis
    print("\n" + "="*60)
    print("CREATING BOOTSTRAP CI ANALYSIS...")
    print("="*60)
    
    # Use the raw entropy data for bootstrap analysis
    bootstrap_fig = create_entropy_bootstrap_ci_figure(entropy_analysis['entropy_data'])

    print(f'Figure S8 saved to ./pngs/figure_S8.png')
    
    # # Create statistical significance matrices
    # print("\n" + "="*60)
    # print("CREATING STATISTICAL MATRICES...")
    # print("="*60)
    
    # p_matrices, effect_matrices = create_entropy_statistical_matrices(entropy_analysis['entropy_data'])
    
    # # Also create the original provider rankings for comparison (optional)
    # provider_rankings = entropy_analysis['provider_rankings']
    # if provider_rankings and False:  # Set to False to skip the old visualization
    #     providers, avg_medians, std_medians, n_values = zip(*provider_rankings)
        
    #     # Create a horizontal bar plot for providers
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
    #     # Provider-level plot
    #     y_pos = np.arange(len(providers))
    #     provider_colors = {'Anthropic': 'orange', 'Google': 'blue', 'OpenAI': 'green', 'Atlas': 'red'}
    #     colors = [provider_colors.get(provider, 'gray') for provider in providers]
        
    #     bars1 = ax1.barh(y_pos, avg_medians, xerr=std_medians, alpha=0.7, color=colors)
        
    #     ax1.set_yticks(y_pos)
    #     ax1.set_yticklabels(providers, fontsize=12)
    #     ax1.invert_yaxis()  # Highest at top
    #     ax1.set_xlabel('Average Median Cumulative Entropy', fontsize=12)
    #     ax1.set_title('Provider Rankings by Average Median Cumulative Entropy\n(Higher = More Exploratory)', fontsize=14)
    #     ax1.grid(axis='x', alpha=0.3)
        
    #     # Add global mean line
    #     global_mean = entropy_analysis['global_stats']['mean']
    #     ax1.axvline(global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Mean ({global_mean:.3f})')
    #     ax1.legend()
        
    #     # Method-level plot (grouped by provider)
    #     method_rankings = entropy_analysis['method_rankings']
    #     if method_rankings:
    #         # Filter to show only top methods per provider (max 3 per provider)
    #         provider_top_methods = defaultdict(list)
    #         for method, avg_med, std_med, n_datasets, provider in method_rankings:
    #             if len(provider_top_methods[provider]) < 3:
    #                 provider_top_methods[provider].append((method, avg_med, std_med, n_datasets))
            
    #         # Flatten and prepare for plotting
    #         plot_methods = []
    #         plot_medians = []
    #         plot_stds = []
    #         plot_colors = []
    #         plot_labels = []
            
    #         for provider in ["Anthropic", "Google", "OpenAI", "Atlas"]:  # Ordered
    #             if provider in provider_top_methods:
    #                 for method, avg_med, std_med, n_datasets in provider_top_methods[provider]:
    #                     plot_methods.append(f"{method}")
    #                     plot_medians.append(avg_med)
    #                     plot_stds.append(std_med)
    #                     plot_colors.append(provider_colors.get(provider, 'gray'))
    #                     plot_labels.append(f"{provider}")
            
    #         y_pos2 = np.arange(len(plot_methods))
    #         bars2 = ax2.barh(y_pos2, plot_medians, xerr=plot_stds, alpha=0.7, color=plot_colors)
            
    #         ax2.set_yticks(y_pos2)
    #         ax2.set_yticklabels([f"{method}" for method in plot_methods], fontsize=10)
    #         ax2.invert_yaxis()
    #         ax2.set_xlabel('Average Median Cumulative Entropy', fontsize=12)
    #         ax2.set_title('Top Methods per Provider\n(Max 3 per Provider)', fontsize=14)
    #         ax2.grid(axis='x', alpha=0.3)
    #         ax2.axvline(global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Mean ({global_mean:.3f})')
    #         ax2.legend()
        
    #     plt.tight_layout()
        
    #     # Save the figure
    #     os.makedirs('./pngs', exist_ok=True)
    #     plt.savefig('./pngs/entropy_provider_rankings.png', dpi=300, bbox_inches='tight')
    #     print("Saved provider rankings visualization to ./pngs/entropy_provider_rankings.png")
        
    #     plt.show()
    
    # print("\n" + "="*60)
    # print("ANALYSIS COMPLETE!")
    # print("="*60)
    # print(f"📊 Total methods analyzed: {len(entropy_analysis['method_rankings'])}")
    # print(f"🏢 Providers analyzed: {len(entropy_analysis['provider_rankings'])}")
    # print(f"📈 Global entropy range: {entropy_analysis['global_stats']['mean']:.3f} ± {entropy_analysis['global_stats']['std']:.3f}")
    
    # # Highlight the most and least exploratory providers
    # if provider_rankings:
    #     most_exploratory_provider = provider_rankings[0]
    #     least_exploratory_provider = provider_rankings[-1]
    #     print(f"🔝 Most exploratory provider: {most_exploratory_provider[0]} ({most_exploratory_provider[1]:.3f})")
    #     print(f"🔻 Least exploratory provider: {least_exploratory_provider[0]} ({least_exploratory_provider[1]:.3f})")
    
    # # Highlight the most and least exploratory individual methods
    # method_rankings = entropy_analysis['method_rankings']
    # if method_rankings:
    #     most_exploratory_method = method_rankings[0]
    #     least_exploratory_method = method_rankings[-1]
    #     print(f"🎯 Most exploratory method: {most_exploratory_method[0]} ({most_exploratory_method[4]}) - {most_exploratory_method[1]:.3f}")
    #     print(f"🎯 Least exploratory method: {least_exploratory_method[0]} ({least_exploratory_method[4]}) - {least_exploratory_method[1]:.3f}")

if __name__ == "__main__":
    main()
