#!/usr/bin/env python3
"""
Statistical analysis comparing all methods (LLM + BO) against random baseline.
Creates tables showing which methods significantly outperform random chance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy import stats

# Import the statistical functions and data loading from figure_5
from figure_5_S12 import (
    get_top_obs_data, get_tracks, dataset_names, dataset_to_obj, model_to_provider,
    dataset_to_color, dataset_order, perform_statistical_analysis, get_provider_data
)

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

def create_methods_vs_random_analysis(top_obs_data, save_path="./"):
    """Create comprehensive analysis of all methods vs random baseline"""
    
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
    
    # Dataset name mapping for clean display
    dataset_name_mapping = {
        'suzuki_doyle': 'Suzuki Yield',
        'suzuki_cernak': 'Suzuki Conversion', 
        'chan_lam_full': 'Chan-Lam',
        'buchwald_hartwig': 'Buchwald-Hartwig',
        'reductive_amination': 'Reductive Amination',
        'alkylation_deprotection': 'Alkylation Deprotection'
    }
    
    # Collect results for each dataset
    all_results = []
    
    for dataset_key in sorted_dataset_names:
        if dataset_key not in filtered_top_obs_data:
            continue
            
        dataset_data = filtered_top_obs_data[dataset_key]
        dataset_display_name = dataset_name_mapping.get(dataset_key, dataset_key.replace('_', ' ').title())
        
        # Find random baseline data
        random_data = None
        for method_key, method_info in dataset_data.items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            if method_name == 'random':
                random_data = method_info['top_obs']
                break
        
        if random_data is None:
            print(f"Warning: No random data found for {dataset_key}")
            continue
        
        # Compare all other methods against random
        for method_key, method_info in dataset_data.items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '').replace('-preview-03-25', '')
            method_name = method_name.replace('-20250514', '').replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '')
            
            # Skip random vs random comparison
            if method_name == 'random':
                continue
            
            method_data = method_info['top_obs']
            
            # Perform statistical test
            p_val, effect_size, result_text = perform_statistical_analysis(method_data, random_data, method_name, 'random')
            
            # Calculate medians and bootstrap CIs
            method_median, method_ci_lower, method_ci_upper = bootstrap_median_ci(method_data)
            random_median, random_ci_lower, random_ci_upper = bootstrap_median_ci(random_data)
            
            # Determine method provider
            provider = 'Other'
            for prov, methods_list in model_to_provider.items():
                if method_name in methods_list:
                    provider = prov
                    break
            
            # Determine significance level
            if p_val is not None:
                if p_val < 0.001:
                    significance = '***'
                    sig_level = 'Highly Significant'
                elif p_val < 0.01:
                    significance = '**'
                    sig_level = 'Very Significant'
                elif p_val < 0.05:
                    significance = '*'
                    sig_level = 'Significant'
                else:
                    significance = 'ns'
                    sig_level = 'Not Significant'
            else:
                significance = 'N/A'
                sig_level = 'N/A'
            
            # Determine practical significance from effect size
            if effect_size is not None:
                abs_effect = abs(effect_size)
                if abs_effect < 0.147:
                    practical_sig = 'Negligible'
                elif abs_effect < 0.33:
                    practical_sig = 'Small'
                elif abs_effect < 0.474:
                    practical_sig = 'Medium'
                else:
                    practical_sig = 'Large'
                
                # Direction
                if effect_size > 0:
                    direction = 'Better than Random'
                else:
                    direction = 'Worse than Random'
            else:
                practical_sig = 'N/A'
                direction = 'N/A'
            
            all_results.append({
                'Dataset': dataset_display_name,
                'Provider': provider,
                'Method': method_name,
                'Method_Median': method_median,
                'Method_CI_Lower': method_ci_lower,
                'Method_CI_Upper': method_ci_upper,
                'Random_Median': random_median,
                'Random_CI_Lower': random_ci_lower,
                'Random_CI_Upper': random_ci_upper,
                'P_Value': p_val,
                'Effect_Size': effect_size,
                'Significance': significance,
                'Sig_Level': sig_level,
                'Practical_Significance': practical_sig,
                'Direction': direction,
                'Performance_Improvement': method_median - random_median if (method_median is not None and random_median is not None) else None
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df

def create_methods_vs_random_tables(results_df, save_path="./pngs/"):
    """Create publication-ready tables showing methods vs random comparisons"""
    
    # Set font
    plt.rcParams['font.family'] = 'SF Pro Display'
    
    # Get unique datasets
    datasets = results_df['Dataset'].unique()
    
    # Create individual table for each dataset
    for dataset in datasets:
        dataset_data = results_df[results_df['Dataset'] == dataset].copy()
        
        # Sort by performance improvement (descending)
        dataset_data = dataset_data.sort_values('Performance_Improvement', ascending=False)
        
        # Create figure for table
        fig, ax = plt.subplots(figsize=(16, max(8, len(dataset_data) * 0.4 + 2)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        
        # Group by provider for better organization
        providers = ['Anthropic', 'Google', 'OpenAI', 'Atlas', 'Other']
        
        for provider in providers:
            provider_data = dataset_data[dataset_data['Provider'] == provider]
            if len(provider_data) == 0:
                continue
            
            # Add provider header
            table_data.append([f'{provider}', '', '', '', '', ''])
            
            # Add methods for this provider
            for _, row in provider_data.iterrows():
                p_val_str = f"{row['P_Value']:.2e}{row['Significance']}" if row['P_Value'] is not None and row['P_Value'] < 0.01 else f"{row['P_Value']:.3f}{row['Significance']}" if row['P_Value'] is not None else 'N/A'
                effect_str = f"{row['Effect_Size']:.3f}" if row['Effect_Size'] is not None else 'N/A'
                improvement_str = f"+{row['Performance_Improvement']:.1f}%" if row['Performance_Improvement'] is not None and row['Performance_Improvement'] > 0 else f"{row['Performance_Improvement']:.1f}%" if row['Performance_Improvement'] is not None else 'N/A'
                
                table_data.append([
                    f"  {row['Method']}",  # Indent method names
                    f"{row['Method_Median']:.1f}%",
                    p_val_str,
                    effect_str,
                    row['Practical_Significance'],
                    improvement_str
                ])
        
        # Create table
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=['Method', 'Median %', 'p-value', 'Effect Size (δ)', 'Practical Sig.', 'vs Random'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            
            # Adjust column widths
            cellDict = table.get_celld()
            col_widths = [0.25, 0.12, 0.15, 0.12, 0.16, 0.12]  # Method, Median, p-value, Effect Size, Practical Sig, vs Random
            
            for i in range(len(table_data) + 1):  # +1 for header row
                for j, width in enumerate(col_widths):
                    cellDict[(i, j)].set_width(width)
            
            # Color coding for significance and effect sizes
            row_idx = 1
            for provider in providers:
                provider_data = dataset_data[dataset_data['Provider'] == provider]
                if len(provider_data) == 0:
                    continue
                
                # Skip provider header row
                row_idx += 1
                
                # Color the method rows for this provider
                for _, row in provider_data.iterrows():
                    # Color p-value based on significance
                    if row['P_Value'] is not None:
                        if row['P_Value'] < 0.001:
                            p_color = '#2b83ba'  # Dark blue (highly significant)
                            p_text_color = 'white'
                        elif row['P_Value'] < 0.01:
                            p_color = '#abdda4'  # Medium green
                            p_text_color = 'black'
                        elif row['P_Value'] < 0.05:
                            p_color = '#fee08b'  # Light yellow
                            p_text_color = 'black'
                        else:
                            p_color = '#d73027'  # Red (not significant)
                            p_text_color = 'white'
                        
                        cellDict[(row_idx, 2)].set_facecolor(p_color)
                        cellDict[(row_idx, 2)].set_text_props(color=p_text_color, weight='bold')
                    
                    # Color effect size based on magnitude and direction
                    if row['Effect_Size'] is not None:
                        effect_val = row['Effect_Size']
                        if effect_val > 0.474:  # Large positive effect
                            effect_color = '#1a9850'  # Dark green
                            effect_text_color = 'white'
                        elif effect_val > 0.33:  # Medium positive effect
                            effect_color = '#91bfdb'  # Light blue
                            effect_text_color = 'black'
                        elif effect_val > 0.147:  # Small positive effect
                            effect_color = '#e6f5d0'  # Very light green
                            effect_text_color = 'black'
                        elif effect_val > -0.147:  # Negligible effect
                            effect_color = '#f7f7f7'  # Light gray
                            effect_text_color = 'black'
                        else:  # Negative effect (worse than random)
                            effect_color = '#fc8d59'  # Orange
                            effect_text_color = 'black'
                        
                        cellDict[(row_idx, 3)].set_facecolor(effect_color)
                        cellDict[(row_idx, 3)].set_text_props(color=effect_text_color, weight='bold')
                    
                    # Color improvement based on value
                    if row['Performance_Improvement'] is not None:
                        improvement = row['Performance_Improvement']
                        if improvement > 10:  # Large improvement
                            imp_color = '#1a9850'  # Dark green
                            imp_text_color = 'white'
                        elif improvement > 5:  # Medium improvement
                            imp_color = '#91bfdb'  # Light blue
                            imp_text_color = 'black'
                        elif improvement > 0:  # Small improvement
                            imp_color = '#e6f5d0'  # Very light green
                            imp_text_color = 'black'
                        else:  # No improvement or worse
                            imp_color = '#fc8d59'  # Orange
                            imp_text_color = 'black'
                        
                        cellDict[(row_idx, 5)].set_facecolor(imp_color)
                        cellDict[(row_idx, 5)].set_text_props(color=imp_text_color, weight='bold')
                    
                    row_idx += 1
            
            # Style header row
            for i in range(6):
                cellDict[(0, i)].set_facecolor('#4a4a4a')
                cellDict[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style provider rows (bold, different background)
            row_idx = 1
            for provider in providers:
                provider_data = dataset_data[dataset_data['Provider'] == provider]
                if len(provider_data) == 0:
                    continue
                
                # Provider header row
                for i in range(6):
                    cellDict[(row_idx, i)].set_facecolor('#D0D0D0')
                    cellDict[(row_idx, i)].set_text_props(weight='bold')
                
                row_idx += 1 + len(provider_data)  # Skip provider methods
        
        # Add title
        plt.suptitle(f'{dataset} - Methods vs Random Baseline Comparison\n'
                    f'Statistical Significance and Effect Sizes', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add legend/notes
        legend_text = ('Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n'
                      'Effect Size: Cliff\'s δ (>0.474 large, >0.33 medium, >0.147 small, <0.147 negligible)\n'
                      'Positive values indicate better performance than random baseline')
        
        plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10, 
                   style='italic', wrap=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        
        # Save figure
        os.makedirs(save_path, exist_ok=True)
        safe_dataset_name = dataset.lower().replace(' ', '_').replace('-', '_')
        filename = f'figure_S13_{safe_dataset_name}.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        
        plt.show()
        plt.close()

def create_methods_vs_random_combined_figure(results_df, save_path="./pngs/"):
    """Create a single 3x2 subplot figure with tables showing methods vs random for each dataset"""
    
    # Set font
    plt.rcParams['font.family'] = 'SF Pro Display'
    
    # Dataset ordering (same as other figures)
    dataset_to_color = {
        'reductive_amination': '#221150',
        'buchwald_hartwig': '#5e177f',
        'chan_lam_full': '#972c7f',
        'suzuki_cernak': '#d3426d',
        'suzuki_doyle': '#f8755c',
        'alkylation_deprotection': '#febb80'
    }
    
    # Dataset name mapping for clean display
    dataset_name_mapping = {
        'Suzuki Yield': 'suzuki_doyle',
        'Suzuki Conversion': 'suzuki_cernak', 
        'Chan-Lam': 'chan_lam_full',
        'Buchwald-Hartwig': 'buchwald_hartwig',
        'Reductive Amination': 'reductive_amination',
        'Alkylation Deprotection': 'alkylation_deprotection'
    }
    
    sorted_dataset_names = sorted(dataset_to_color.keys(), 
                                 key=lambda x: list(dataset_to_color.values()).index(dataset_to_color[x]), 
                                 reverse=True)
    
    # Convert back to display names
    display_names = []
    for key in sorted_dataset_names:
        for display_name, key_name in dataset_name_mapping.items():
            if key_name == key:
                display_names.append(display_name)
                break
    
    # Create 3x2 subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes_flat = axes.flatten()
    
    for idx, display_name in enumerate(display_names):
        if idx >= 6:  # Only handle first 6 datasets
            break
            
        ax = axes_flat[idx]
        ax.axis('off')  # Turn off axis for table
        
        # Get data for this dataset
        dataset_data = results_df[results_df['Dataset'] == display_name].copy()
        
        if len(dataset_data) == 0:
            ax.text(0.5, 0.5, f'No data for {display_name}', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            continue
        
        # Sort by performance improvement (descending)
        dataset_data = dataset_data.sort_values('Performance_Improvement', ascending=False)
        
        # Prepare table data - show only top methods to fit in space
        table_data = []
        
        # Group by provider but limit to top methods
        providers = ['Anthropic', 'Google', 'OpenAI', 'Atlas']
        max_methods_per_provider = 2  # Limit to fit in space
        
        for provider in providers:
            provider_data = dataset_data[dataset_data['Provider'] == provider].head(max_methods_per_provider)
            if len(provider_data) == 0:
                continue
                
            # Add provider header row
            table_data.append([f'{provider}', '', '', ''])
            
            # Add methods for this provider
            for _, row in provider_data.iterrows():
                # Format p-value
                if row['P_Value'] is not None:
                    if row['P_Value'] < 0.001:
                        p_val_str = f"{row['P_Value']:.2e}***"
                    elif row['P_Value'] < 0.01:
                        p_val_str = f"{row['P_Value']:.3f}**"
                    elif row['P_Value'] < 0.05:
                        p_val_str = f"{row['P_Value']:.3f}*"
                    else:
                        p_val_str = f"{row['P_Value']:.3f}"
                else:
                    p_val_str = 'N/A'
                
                # Format effect size
                effect_str = f"{row['Effect_Size']:.2f}" if row['Effect_Size'] is not None else 'N/A'
                
                # Format improvement
                if row['Performance_Improvement'] is not None:
                    if row['Performance_Improvement'] > 0:
                        improvement_str = f"+{row['Performance_Improvement']:.1f}%"
                    else:
                        improvement_str = f"{row['Performance_Improvement']:.1f}%"
                else:
                    improvement_str = 'N/A'
                
                table_data.append([
                    f"  {row['Method'][:15]}{'...' if len(row['Method']) > 15 else ''}",  # Truncate long names
                    p_val_str,
                    effect_str,
                    improvement_str
                ])
        
        # Create table
        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=['Method', 'p-value', 'Effect δ', 'vs Random'],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            
            # Adjust column widths
            cellDict = table.get_celld()
            col_widths = [0.4, 0.2, 0.2, 0.2]  # Method, p-value, Effect Size, vs Random
            
            for i in range(len(table_data) + 1):  # +1 for header row
                for j, width in enumerate(col_widths):
                    cellDict[(i, j)].set_width(width)
            
            # Apply color coding (simplified for space)
            row_idx = 1
            for provider in providers:
                provider_data = dataset_data[dataset_data['Provider'] == provider].head(max_methods_per_provider)
                if len(provider_data) == 0:
                    continue
                
                # Skip provider header row
                row_idx += 1
                
                # Color the method rows for this provider
                for _, row in provider_data.iterrows():
                    # Color p-value based on significance
                    if row['P_Value'] is not None:
                        if row['P_Value'] < 0.001:
                            p_color = '#2b83ba'  # Dark blue
                            p_text_color = 'white'
                        elif row['P_Value'] < 0.01:
                            p_color = '#abdda4'  # Medium green
                            p_text_color = 'black'
                        elif row['P_Value'] < 0.05:
                            p_color = '#fee08b'  # Light yellow
                            p_text_color = 'black'
                        else:
                            p_color = '#d73027'  # Red
                            p_text_color = 'white'
                        
                        cellDict[(row_idx, 1)].set_facecolor(p_color)
                        cellDict[(row_idx, 1)].set_text_props(color=p_text_color, weight='bold')
                    
                    # Color effect size
                    if row['Effect_Size'] is not None:
                        effect_val = row['Effect_Size']
                        if effect_val > 0.33:  # Medium+ positive effect
                            effect_color = '#1a9850'  # Dark green
                            effect_text_color = 'white'
                        elif effect_val > 0.147:  # Small positive effect
                            effect_color = '#91bfdb'  # Light blue
                            effect_text_color = 'black'
                        elif effect_val > -0.147:  # Negligible effect
                            effect_color = '#f7f7f7'  # Light gray
                            effect_text_color = 'black'
                        else:  # Negative effect
                            effect_color = '#fc8d59'  # Orange
                            effect_text_color = 'black'
                        
                        cellDict[(row_idx, 2)].set_facecolor(effect_color)
                        cellDict[(row_idx, 2)].set_text_props(color=effect_text_color, weight='bold')
                    
                    # Color improvement
                    if row['Performance_Improvement'] is not None:
                        improvement = row['Performance_Improvement']
                        if improvement > 10:  # Large improvement
                            imp_color = '#1a9850'  # Dark green
                            imp_text_color = 'white'
                        elif improvement > 5:  # Medium improvement
                            imp_color = '#91bfdb'  # Light blue
                            imp_text_color = 'black'
                        elif improvement > 0:  # Small improvement
                            imp_color = '#e6f5d0'  # Very light green
                            imp_text_color = 'black'
                        else:  # No improvement or worse
                            imp_color = '#fc8d59'  # Orange
                            imp_text_color = 'black'
                        
                        cellDict[(row_idx, 3)].set_facecolor(imp_color)
                        cellDict[(row_idx, 3)].set_text_props(color=imp_text_color, weight='bold')
                    
                    row_idx += 1
            
            # Style header row
            for i in range(4):
                cellDict[(0, i)].set_facecolor('#4a4a4a')
                cellDict[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style provider rows
            row_idx = 1
            for provider in providers:
                provider_data = dataset_data[dataset_data['Provider'] == provider].head(max_methods_per_provider)
                if len(provider_data) == 0:
                    continue
                
                # Provider header row
                for i in range(4):
                    cellDict[(row_idx, i)].set_facecolor('#D0D0D0')
                    cellDict[(row_idx, i)].set_text_props(weight='bold')
                
                row_idx += 1 + len(provider_data)
        
        # Add dataset title
        ax.text(0.5, 1.02, display_name, ha='center', va='bottom', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Hide any unused subplots
    for idx in range(len(display_names), 6):
        axes_flat[idx].set_visible(False)
    
    # Add main title
    plt.suptitle('Methods vs Random Baseline - Statistical Comparison\n'
                '(Top 2 methods per provider shown)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend at the bottom
    legend_text = ('Significance: *** p<0.001, ** p<0.01, * p<0.05  |  '
                  'Effect Size: Cliff\'s δ (>0.33 medium+, >0.147 small, <0.147 negligible)  |  '
                  'Positive values = better than random')
    
    plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10, 
               style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08)
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, 'methods_vs_random_combined_table.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved combined table figure: {filepath}")
    
    plt.show()
    plt.close()
    
    return fig

def create_methods_vs_random_summary_table(results_df, save_path="./"):
    """Create a summary table across all datasets"""
    
    # Calculate summary statistics per method across all datasets
    method_summary = []
    
    for method in results_df['Method'].unique():
        method_data = results_df[results_df['Method'] == method]
        
        # Get provider
        provider = method_data['Provider'].iloc[0]
        
        # Calculate statistics
        significant_datasets = len(method_data[method_data['P_Value'] < 0.05])
        total_datasets = len(method_data)
        avg_effect_size = method_data['Effect_Size'].mean()
        avg_improvement = method_data['Performance_Improvement'].mean()
        
        # Count effect size categories
        large_effects = len(method_data[method_data['Effect_Size'] > 0.474])
        medium_effects = len(method_data[(method_data['Effect_Size'] > 0.33) & (method_data['Effect_Size'] <= 0.474)])
        small_effects = len(method_data[(method_data['Effect_Size'] > 0.147) & (method_data['Effect_Size'] <= 0.33)])
        
        method_summary.append({
            'Method': method,
            'Provider': provider,
            'Significant_Datasets': significant_datasets,
            'Total_Datasets': total_datasets,
            'Significance_Rate': significant_datasets / total_datasets,
            'Avg_Effect_Size': avg_effect_size,
            'Avg_Improvement': avg_improvement,
            'Large_Effects': large_effects,
            'Medium_Effects': medium_effects,
            'Small_Effects': small_effects
        })
    
    summary_df = pd.DataFrame(method_summary)
    summary_df = summary_df.sort_values('Significance_Rate', ascending=False)
    
    # Save as CSV
    os.makedirs(save_path, exist_ok=True)
    summary_df.to_csv(os.path.join(save_path, 'methods_vs_random_summary.csv'), index=False)
    results_df.to_csv(os.path.join(save_path, 'methods_vs_random_detailed.csv'), index=False)
    
    print(f"Summary statistics saved to {save_path}")
    print(f"  - methods_vs_random_summary.csv (method-level summary)")
    print(f"  - methods_vs_random_detailed.csv (full detailed results)")
    
    return summary_df

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
    
    print("Creating methods vs random analysis...")
    results_df = create_methods_vs_random_analysis(top_obs_data)
    
    print(f"Analyzed {len(results_df)} method-dataset combinations")
    
    # print("\nCreating combined table figure...")
    # create_methods_vs_random_combined_figure(results_df, save_path="./pngs/")
    
    print("\nCreating individual dataset tables...")
    create_methods_vs_random_tables(results_df, save_path="./pngs/")
    
    # print("\nCreating summary statistics...")
    # summary_df = create_methods_vs_random_summary_table(results_df, save_path="./")
    
    # print("\nAnalysis complete!")
    # print(f"📊 Total comparisons: {len(results_df)}")
    # print(f"📈 Significant improvements over random: {len(results_df[results_df['P_Value'] < 0.05])}")
    # print(f"📉 Methods worse than random: {len(results_df[results_df['Effect_Size'] < 0])}")
    
    # # Show top performers
    # top_performers = results_df.nlargest(5, 'Performance_Improvement')
    # print(f"\n🏆 Top 5 improvements over random:")
    # for _, row in top_performers.iterrows():
    #     print(f"   {row['Method']} ({row['Dataset']}): +{row['Performance_Improvement']:.1f}% (p={row['P_Value']:.2e})")

    print(f'Figure S13 saved to ./pngs/figure_S13_Alkylation_Deprotection.png')
    print(f'Figure S13 saved to ./pngs/figure_S13_Chan_Lam_Full.png')
    print(f'Figure S13 saved to ./pngs/figure_S13_Buchwald_Hartwig.png')
    print(f'Figure S13 saved to ./pngs/figure_S13_Reductive_Amination.png')
    print(f'Figure S13 saved to ./pngs/figure_S13_Suzuki_Doyle.png')
    print(f'Figure S13 saved to ./pngs/figure_S13_Suzuki_Cernak.png')
