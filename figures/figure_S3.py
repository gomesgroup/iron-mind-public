import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Import the statistical functions and data loading from figure_5
from figure_5_S12 import (
    get_top_obs_data, get_tracks, dataset_names, dataset_to_obj, model_to_provider,
    dataset_to_color, dataset_order, perform_statistical_analysis, get_provider_data,
    find_best_llm_vs_best_bo_per_dataset
)

def get_overall_best_method_per_dataset(top_obs_data):
    """Find the single best performing method (LLM or BO) for each dataset with proper filtering"""
    best_methods = {}
    
    for dataset_name, dataset_data in top_obs_data.items():
        method_performances = {}
        
        # Calculate median performance for each method with filtering
        for method_key, method_info in dataset_data.items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
            
            median_perf = np.median(method_info['top_obs'])
            method_performances[method_name] = {
                'median': median_perf,
                'data': method_info['top_obs'],
                'full_key': method_key
            }
        
        if not method_performances:
            continue
            
        # Find the best method (highest median)
        best_method_name = max(method_performances.keys(), key=lambda x: method_performances[x]['median'])
        best_method_info = method_performances[best_method_name]
        
        best_methods[dataset_name] = {
            'method': best_method_name,
            'median': best_method_info['median'],
            'data': best_method_info['data']
        }
    
    return best_methods

def bootstrap_median_ci(data, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for median"""
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    data = np.array(data)
    n = len(data)
    
    # Generate bootstrap samples
    bootstrap_medians = []
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))
    
    bootstrap_medians = np.array(bootstrap_medians)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_medians, lower_percentile)
    ci_upper = np.percentile(bootstrap_medians, upper_percentile)
    median = np.median(data)
    
    return median, ci_lower, ci_upper

def create_bootstrap_ci_table(top_obs_data, save_path="./"):
    """Create a table of bootstrap confidence intervals for medians"""
    
    # Apply same method filtering as figure_5.py
    filtered_top_obs_data = {}
    for dataset_name, dataset_data in top_obs_data.items():
        filtered_top_obs_data[dataset_name] = {}
        for key, method_info in dataset_data.items():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
            # Apply filtering - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
                
            filtered_top_obs_data[dataset_name][key] = method_info
    
    # Sort datasets by color order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Collect all methods across all datasets
    all_methods = set()
    for dataset_data in filtered_top_obs_data.values():
        for key in dataset_data.keys():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            all_methods.add(method_name)
    
    # Sort methods by provider for better organization
    provider_methods = {}
    for provider, methods_list in model_to_provider.items():
        provider_methods[provider] = [m for m in methods_list if m in all_methods]
    
    # Create table data
    table_data = []
    
    for dataset_key in sorted_dataset_names:
        if dataset_key not in filtered_top_obs_data:
            continue
            
        dataset_data = filtered_top_obs_data[dataset_key]
        dataset_name = dataset_key.replace('_', ' ').title()
        
        # Group methods by provider
        for provider in ['Anthropic', 'Google', 'OpenAI', 'Atlas', 'Random']:
            if provider not in provider_methods or not provider_methods[provider]:
                continue
                
            for method_name in sorted(provider_methods[provider]):
                # Find data for this method in this dataset
                method_data = None
                for key, data in dataset_data.items():
                    key_method = key.split('/')[-1]
                    key_method = key_method.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
                    key_method = key_method.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
                    key_method = key_method.replace('-preview-06-17', '')
                    key_method = key_method.replace('-20250805', '')
                    
                    if key_method == method_name:
                        method_data = data['top_obs']
                        break
                
                if method_data is not None:
                    median, ci_lower, ci_upper = bootstrap_median_ci(method_data)
                    
                    table_data.append({
                        'Dataset': dataset_name,
                        'Provider': provider,
                        'Method': method_name,
                        'Median': f"{median:.1f}",
                        'CI_Lower': f"{ci_lower:.1f}",
                        'CI_Upper': f"{ci_upper:.1f}",
                        'CI_Range': f"[{ci_lower:.1f}, {ci_upper:.1f}]",
                        'CI_Width': f"{ci_upper - ci_lower:.1f}"
                    })
    
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(table_data)
    
    return df

def create_bootstrap_ci_figure(top_obs_data, save_path="./pngs/"):
    """Create a 3x2 subplot figure with tables showing bootstrap CIs for each dataset"""
    
    # Get the CI data
    ci_df = create_bootstrap_ci_table(top_obs_data, save_path="./")
    
    if len(ci_df) == 0:
        print("No data available for bootstrap CI figure")
        return None
    
    # Set font
    plt.rcParams['font.family'] = 'SF Pro Display'
    
    # Sort datasets by color order (same as other figures)
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Create 3x2 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.flatten()
    
    for idx, dataset_key in enumerate(sorted_dataset_names):
        if idx >= 6:  # Only handle first 6 datasets
            break
            
        ax = axes_flat[idx]
        ax.axis('off')  # Turn off axis for table
        
        # Convert dataset key to clean display name
        dataset_name_mapping = {
            'suzuki_doyle': 'Suzuki Yield',
            'suzuki_cernak': 'Suzuki Conversion', 
            'chan_lam_full': 'Chan-Lam',
            'buchwald_hartwig': 'Buchwald-Hartwig',
            'reductive_amination': 'Reductive Amination',
            'alkylation_deprotection': 'Alkylation Deprotection'
        }
        
        display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
        # Use the original format for data lookup
        dataset_name = dataset_key.replace('_', ' ').title()
        dataset_data = ci_df[ci_df['Dataset'] == dataset_name]
        
        if len(dataset_data) == 0:
            ax.text(0.5, 0.5, f'No data for {dataset_name}', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue
        
        # Prepare table data
        table_data = []
        
        # Group by provider for better organization
        providers = ['Anthropic', 'Google', 'OpenAI', 'Atlas', 'Random']
        
        for provider in providers:
            provider_data = dataset_data[dataset_data['Provider'] == provider]
            if len(provider_data) == 0:
                continue
                
            # Add provider header row
            table_data.append([f'{provider}', '', '', ''])
            
            # Add methods for this provider
            for _, row in provider_data.iterrows():
                method_name = row['Method']
                # Keep full method name since we're making the column wider
                
                table_data.append([
                    f'  {method_name}',  # Indent method names
                    f"{row['Median']}%",
                    f"{row['CI_Lower']}%",
                    f"{row['CI_Upper']}%"
                ])
        
        # Create table
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=['Method', 'Median', 'CI Lower', 'CI Upper'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style the table
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
            
            # Apply systematic color coding to percentage values
            def get_color_for_percentage(value_str):
                """Convert percentage string to color - blue (0%) to red (100%)"""
                try:
                    # Extract numeric value from string like "95.2%"
                    value = float(value_str.replace('%', ''))
                    # Clamp value between 0 and 100
                    value = max(0, min(100, value))
                    # Create color gradient: blue (0) to red (100)
                    # Blue component decreases as value increases
                    blue = (100 - value) / 100
                    # Red component increases as value increases  
                    red = value / 100
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
                    median_color = get_color_for_percentage(row['Median'])
                    cellDict[(row_idx, 1)].set_text_props(color=median_color)
                    
                    # Color CI Lower value (column 2)
                    ci_lower_color = get_color_for_percentage(row['CI_Lower'])
                    cellDict[(row_idx, 2)].set_text_props(color=ci_lower_color)
                    
                    # Color CI Upper value (column 3)
                    ci_upper_color = get_color_for_percentage(row['CI_Upper'])
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
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'figure_S3.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_statistical_matrices(top_obs_data, save_path="./"):
    """Create statistical comparison matrices for LLM vs BO methods"""
    
    # Apply same method filtering as figure_5.py
    filtered_top_obs_data = {}
    for dataset_name, dataset_data in top_obs_data.items():
        filtered_top_obs_data[dataset_name] = {}
        for key, method_info in dataset_data.items():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
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
    
    # Create matrices for each dataset
    p_value_matrices = {}
    effect_size_matrices = {}
    
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
            method_name = method_name.replace('-20250805', '')
            
            # Classify as LLM or BO
            is_llm = any(method_name in provider_methods for provider, provider_methods in model_to_provider.items() 
                        if provider in ['Anthropic', 'Google', 'OpenAI'])
            is_bo = method_name in model_to_provider['Atlas']
            
            if is_llm:
                llm_methods[method_name] = method_info['top_obs']
            elif is_bo:
                bo_methods[method_name] = method_info['top_obs']
        
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
    create_matrix_plots(p_value_matrices, effect_size_matrices, sorted_dataset_names, save_path)
    
    return p_value_matrices, effect_size_matrices

def create_matrix_plots(p_value_matrices, effect_size_matrices, sorted_dataset_names, save_path):
    """Create 3x2 subplot matrices for statistical comparisons"""
    
    # Create two figures: one for p-values, one for effect sizes
    
    # Figure 1: Wilcoxon p-values
    fig1, axes1 = plt.subplots(3, 2, figsize=(18, 22))
    
    # Figure 2: Cliff's delta effect sizes
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 22))
    
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
        
        dataset_name = dataset_key.replace('_', ' ').title()
        
        # Get data
        p_data = p_value_matrices[dataset_key]
        effect_data = effect_size_matrices[dataset_key]
        
        p_matrix = p_data['matrix']
        effect_matrix = effect_data['matrix']
        llm_names = p_data['llm_names']
        bo_names = p_data['bo_names']
        
        # Plot 1: P-values with significance coloring
        ax1 = axes1_flat[idx]
        
        # Create custom colormap for p-values with better colors
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
        
        ax1.set_title(dataset_name, fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(bo_names)))
        ax1.set_yticks(range(len(llm_names)))
        ax1.set_xticklabels(bo_names, rotation=0, ha='center')
        ax1.set_yticklabels(llm_names)
        ax1.set_xlabel('BO Methods', fontweight='bold')
        ax1.set_ylabel('LLM Methods', fontweight='bold')
        
        # Plot 2: Effect sizes
        ax2 = axes2_flat[idx]
        
        # Use diverging colormap for effect sizes (centered at 0)
        vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()))
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
        
        ax2.set_title(dataset_name, fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(bo_names)))
        ax2.set_yticks(range(len(llm_names)))
        ax2.set_xticklabels(bo_names, rotation=0, ha='center')
        ax2.set_yticklabels(llm_names)
        ax2.set_xlabel('BO Methods', fontweight='bold')
        ax2.set_ylabel('LLM Methods', fontweight='bold')
    
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
    os.makedirs(save_path, exist_ok=True)
    fig1.savefig(os.path.join(save_path, 'statistical_matrices_pvalues.png'), 
                dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(save_path, 'statistical_matrices_effect_sizes.png'), 
                dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig1, fig2

def create_statistical_summary_markdown(top_obs_data, save_path="./"):
    """Create comprehensive statistical summary as markdown tables"""
    
    # Apply same method filtering as figure_5.py
    filtered_top_obs_data = {}
    for dataset_name, dataset_data in top_obs_data.items():
        filtered_top_obs_data[dataset_name] = {}
        for key, method_info in dataset_data.items():
            method_name = key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            method_name = method_name.replace('-20250805', '')
            
            # Apply same filtering as figure_5.py - exclude gpt-4.1 with nano
            if 'gpt-4.1' in method_name and 'nano' in method_name:
                continue
                
            filtered_top_obs_data[dataset_name][key] = method_info
    
    # Get all statistical analyses using filtered data
    best_comparisons = find_best_llm_vs_best_bo_per_dataset(filtered_top_obs_data)
    overall_best_methods = get_overall_best_method_per_dataset(top_obs_data)  # Use original data, filtering is done inside
    
    # Sort datasets by color order
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    markdown_content = []
    
    # Header
    markdown_content.append("# Statistical Analysis Summary\n")
    markdown_content.append("*Focused statistical comparison of LLM vs BO optimization methods*\n")
    markdown_content.append("*Note: Excludes filtered methods (e.g., gpt-4.1-nano) consistent with figure generation*\n")
    
    # Table 1: Best LLM vs Best BO Comparison
    markdown_content.append("## 1. Best LLM vs Best BO Head-to-Head Comparison\n")
    markdown_content.append("| Dataset | Best LLM | LLM % | Best BO | BO % | p-value | Cliff's δ | Winner |")
    markdown_content.append("|---------|----------|-------|---------|------|---------|-----------|--------|")
    
    for dataset_key in sorted_dataset_names:
        dataset_name = dataset_key.replace('_', ' ').title()
        if dataset_key in best_comparisons:
            comp = best_comparisons[dataset_key]
            
            if comp['p_value'] is not None:
                if comp['p_value'] < 0.001:
                    p_str = f"{comp['p_value']:.1e}***"
                elif comp['p_value'] < 0.01:
                    p_str = f"{comp['p_value']:.3f}**"
                elif comp['p_value'] < 0.05:
                    p_str = f"{comp['p_value']:.3f}*"
                else:
                    p_str = f"{comp['p_value']:.3f}"
                
                winner = "🤖 LLM" if comp['effect_size'] > 0 else "🔬 BO"
                effect_str = f"{comp['effect_size']:.3f}"
            else:
                p_str = 'N/A'
                winner = 'N/A'
                effect_str = 'N/A'
            
            markdown_content.append(
                f"| {dataset_name} | {comp['best_llm']} | {comp['llm_median']:.1f}% | "
                f"{comp['best_bo']} | {comp['bo_median']:.1f}% | {p_str} | {effect_str} | {winner} |"
            )
    
    markdown_content.append("")
    
    # Table 2: Overall Best Method per Dataset
    markdown_content.append("## 2. Overall Best Method per Dataset\n")
    markdown_content.append("*Shows the single highest-performing method (LLM or BO) across all methods for each dataset.*\n")
    markdown_content.append("| Dataset | Best Method | Median % | Method Type |")
    markdown_content.append("|---------|-------------|----------|-------------|")
    
    for dataset_key in sorted_dataset_names:
        dataset_name = dataset_key.replace('_', ' ').title()
        if dataset_key in overall_best_methods:
            best_info = overall_best_methods[dataset_key]
            method_name = best_info['method']
            
            # Determine method type
            is_llm = any(method_name in provider_methods for provider, provider_methods in model_to_provider.items() 
                        if provider in ['Anthropic', 'Google', 'OpenAI'])
            is_bo = method_name in model_to_provider['Atlas']
            is_random = method_name == 'random'
            
            if is_llm:
                method_type = "🤖 LLM"
            elif is_bo:
                method_type = "🔬 BO"
            elif is_random:
                method_type = "🎲 Random"
            else:
                method_type = "❓ Other"
            
            markdown_content.append(
                f"| {dataset_name} | **{method_name}** | {best_info['median']:.1f}% | {method_type} |"
            )
    
    markdown_content.append("")
    
    # Table 3: Provider-level LLM vs BO Analysis
    markdown_content.append("## 3. Provider-Level Analysis (All Methods)\n")
    markdown_content.append("| Dataset | All LLM % | All BO % | p-value | Cliff's δ | Random % |")
    markdown_content.append("|---------|-----------|----------|---------|-----------|----------|")
    
    for dataset_key in sorted_dataset_names:
        dataset_name = dataset_key.replace('_', ' ').title()
        
        # Get all LLM vs all BO data
        llm_data, bo_data, random_data = get_provider_data(filtered_top_obs_data, dataset_key)
        
        if len(llm_data) > 0 and len(bo_data) > 0:
            p_val, effect_size, result_text = perform_statistical_analysis(llm_data, bo_data, "LLM", "BO")
            
            if p_val is not None:
                if p_val < 0.001:
                    p_str = f"{p_val:.1e}***"
                elif p_val < 0.01:
                    p_str = f"{p_val:.3f}**"
                elif p_val < 0.05:
                    p_str = f"{p_val:.3f}*"
                else:
                    p_str = f"{p_val:.3f}"
                
                effect_str = f"{effect_size:.3f}"
                llm_median = f"{np.median(llm_data):.1f}%"
                bo_median = f"{np.median(bo_data):.1f}%"
            else:
                p_str = 'N/A'
                effect_str = 'N/A'
                llm_median = 'N/A'
                bo_median = 'N/A'
        else:
            p_str = 'N/A'
            effect_str = 'N/A'
            llm_median = 'N/A'
            bo_median = 'N/A'
        
        random_median = f"{np.median(random_data):.1f}%" if len(random_data) > 0 else 'N/A'
        
        markdown_content.append(
            f"| {dataset_name} | {llm_median} | {bo_median} | {p_str} | {effect_str} | {random_median} |"
        )
    
    markdown_content.append("")
    
    # Summary Statistics
    markdown_content.append("## 4. Summary Statistics\n")
    
    # Calculate summary statistics
    all_p_values = []
    all_effect_sizes = []
    significant_datasets = []
    
    for dataset_key in sorted_dataset_names:
        llm_data, bo_data, random_data = get_provider_data(filtered_top_obs_data, dataset_key)
        if len(llm_data) > 0 and len(bo_data) > 0:
            p_val, effect_size, _ = perform_statistical_analysis(llm_data, bo_data, "LLM", "BO")
            if p_val is not None:
                all_p_values.append(p_val)
                all_effect_sizes.append(effect_size)
                if p_val < 0.05:
                    significant_datasets.append(dataset_key)
    
    # Method wins summary
    method_wins = {}
    provider_wins = {'LLM': 0, 'BO': 0, 'Random': 0, 'Other': 0}
    
    for dataset_key, best_info in overall_best_methods.items():
        method = best_info['method']
        if method not in method_wins:
            method_wins[method] = 0
        method_wins[method] += 1
        
        # Count by provider type
        is_llm = any(method in provider_methods for provider, provider_methods in model_to_provider.items() 
                    if provider in ['Anthropic', 'Google', 'OpenAI'])
        is_bo = method in model_to_provider['Atlas']
        is_random = method == 'random'
        
        if is_llm:
            provider_wins['LLM'] += 1
        elif is_bo:
            provider_wins['BO'] += 1
        elif is_random:
            provider_wins['Random'] += 1
        else:
            provider_wins['Other'] += 1
    
    # Create summary data
    bonferroni_alpha = 0.05 / len(all_p_values) if all_p_values else 0
    bonferroni_significant = sum(p < bonferroni_alpha for p in all_p_values) if all_p_values else 0
    
    markdown_content.append("### Overall Statistical Results")
    markdown_content.append("| Metric | Value |")
    markdown_content.append("|--------|-------|")
    markdown_content.append(f"| Total Datasets | {len(sorted_dataset_names)} |")
    markdown_content.append(f"| Significant Differences (p<0.05) | {len(significant_datasets)}/{len(all_p_values)} |")
    markdown_content.append(f"| Bonferroni Significant (α={bonferroni_alpha:.4f}) | {bonferroni_significant}/{len(all_p_values)} |")
    if all_effect_sizes:
        markdown_content.append(f"| Mean Effect Size (Cliff's δ) | {np.mean(all_effect_sizes):.3f} |")
        markdown_content.append(f"| Effect Size Range | [{np.min(all_effect_sizes):.3f}, {np.max(all_effect_sizes):.3f}] |")
    
    markdown_content.append("")
    markdown_content.append("### Provider Type Winners")
    markdown_content.append("| Provider Type | Wins |")
    markdown_content.append("|---------------|------|")
    
    for provider_type, count in provider_wins.items():
        if count > 0:
            emoji = {"LLM": "🤖", "BO": "🔬", "Random": "🎲", "Other": "❓"}[provider_type]
            markdown_content.append(f"| {emoji} {provider_type} | {count}/{len(overall_best_methods)} datasets |")
    
    markdown_content.append("")
    markdown_content.append("### Individual Method Winners")
    markdown_content.append("| Method | Wins |")
    markdown_content.append("|--------|------|")
    
    top_methods = sorted(method_wins.items(), key=lambda x: x[1], reverse=True)
    for method, count in top_methods:
        markdown_content.append(f"| {method} | {count}/{len(overall_best_methods)} datasets |")
    
    # Add interpretation notes
    markdown_content.append("")
    markdown_content.append("## Notes")
    markdown_content.append("- **Significance levels**: `*` p<0.05, `**` p<0.01, `***` p<0.001")
    markdown_content.append("- **Cliff's δ interpretation**: |δ| < 0.147 (negligible), < 0.33 (small), < 0.474 (medium), ≥ 0.474 (large)")
    markdown_content.append("- **Positive δ**: First method outperforms second method")
    markdown_content.append("- **Negative δ**: Second method outperforms first method")
    markdown_content.append("- **Bonferroni correction**: Applied for multiple comparisons across datasets")
    
    # Save markdown file
    markdown_text = '\n'.join(markdown_content)
    
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'statistical_summary.md'), 'w') as f:
        f.write(markdown_text)
    
    return markdown_text

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
    
    ci_fig = create_bootstrap_ci_figure(top_obs_data, save_path="./pngs/")
    
    print(f'Figure S3 saved to ./pngs/figure_S3.png')
    
