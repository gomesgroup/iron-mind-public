#!/usr/bin/env python3
"""
Figure S14: Chan Lam LLM vs BO comparison (with and without paper)
Includes entropy analysis, performance analysis with all aggregation strategies,
and Cliff's delta statistical comparisons for both with-paper and without-paper LLM runs.
"""

import pandas as pd
import glob, json, os
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from olympus.datasets.dataset import Dataset
from scipy import stats

# Set font to sf pro
plt.rcParams['font.family'] = 'SF Pro Display'

# Import functions from other figure scripts
from figure_5_S12 import perform_statistical_analysis, cliffs_delta
from figure_7_S10 import (
    compute_individual_run_cumulative_entropy,
    compute_individual_run_parameter_entropies
)

# Dataset configuration - focus on Chan Lam only
dataset_names = ['Chan_Lam_Full']

dataset_to_obj = {
    'Chan_Lam_Full': {
        'objectives': ['desired_yield', 'undesired_yield'],
        'transform': 'weighted_selectivity',  # (desired/(desired + undesired)) * desired
        'order': [0, 1],  # desired, undesired
        'aggregation': 'min'  # Default, but we'll test all three
    }
}

model_to_provider = {
    "Anthropic": [
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
    'chan_lam_full': '#0071b2',     # Dark Orange
}

def get_objective_value_with_aggregation(group, dataset_config, aggregation_method):
    """
    Calculate Chan Lam weighted selectivity with specified aggregation
    
    Args:
        group: pandas group containing measurements
        dataset_config: dataset configuration
        aggregation_method: 'min', 'mean', or 'max'
    """
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
    
    # Apply specified aggregation
    if weighted_selectivities:
        if aggregation_method == 'min':
            return np.min(weighted_selectivities)
        elif aggregation_method == 'mean':
            return np.mean(weighted_selectivities)
        elif aggregation_method == 'max':
            return np.max(weighted_selectivities)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    else:
        return 0.0

def get_tracks_with_aggregation(path, dataset_name, aggregation_method='min', bo=False, n_tracks=None, track_size=None):
    """
    Extract optimization tracks with specified aggregation method
    """
    run_dirs = [os.path.join(path, d) for d in os.listdir(path) if 'run_' in d]
    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]))
    tracks = []
    
    dataset_config = dataset_to_obj[dataset_name]
    
    for rd in run_dirs:
        seen_path = os.path.join(rd, 'seen_observations.json')
        objective_values = []
        
        if os.path.exists(seen_path):
            with open(seen_path, 'r') as f:
                seen_data = json.load(f)
            
            # Special handling for Chan_Lam_Full
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
            
            # Process groups and extract objectives with specified aggregation
            for name, group in param_groups:
                try:
                    obj_value = get_objective_value_with_aggregation(group, dataset_config, aggregation_method)
                    if np.isnan(obj_value):
                        objective_values.append(0)
                    else:
                        objective_values.append(obj_value)
                except (KeyError, TypeError, ZeroDivisionError):
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
    
    return tracks

def get_performance_data_all_aggregations(path_dict):
    """Extract performance data with all three aggregation methods"""
    performance_data = {
        'min': {},
        'mean': {},
        'max': {}
    }
    
    for aggregation_method in ['min', 'mean', 'max']:
        performance_data[aggregation_method]['chan_lam_full'] = {}
        
        for method_key, _ in path_dict['Chan_Lam_Full'].items():
            # Get tracks with this aggregation method
            is_bo = 'bayesian' in method_key
            tracks = get_tracks_with_aggregation(
                method_key, 'Chan_Lam_Full', 
                aggregation_method=aggregation_method,
                bo=is_bo, n_tracks=20, track_size=20
            )
            
            if tracks is not None:
                # Get the max in each track (final performance)
                top_obs = [float(np.max(track)) for track in tracks if len(track) > 0]
                if top_obs:
                    performance_data[aggregation_method]['chan_lam_full'][method_key] = {
                        'top_obs': top_obs,
                        'median': float(np.median(top_obs)),
                        'q1': float(np.percentile(top_obs, 25)),
                        'q3': float(np.percentile(top_obs, 75))
                    }
    
    return performance_data

def get_entropy_data(path_dict, dataset_param_options):
    """Extract entropy data for Chan Lam methods"""
    entropy_data = {}
    entropy_data['chan_lam_full'] = {}
    param_options = dataset_param_options['Chan_Lam_Full']
    
    for method_path, _ in path_dict['Chan_Lam_Full'].items():
        entropy_data['chan_lam_full'][method_path] = {
            'cumulative_entropies': [], 
            'median': None, 
            'q1': None, 
            'q3': None
        }
        
        # Get all run directories for this method
        runs = glob.glob(method_path + "/run_*/")
        run_entropies = []
        
        for run_dir in runs:
            try:
                # Load seen observations for this run
                with open(os.path.join(run_dir, "seen_observations.json"), "r") as f:
                    seen_observations = json.load(f)
                
                df = pd.DataFrame(seen_observations)
                
                # Group observations similar to figure_7.py approach
                if "llm" in method_path:
                    group_cols = ["reasoning", "explanation"]
                elif "bayesian" in method_path:
                    all_cols = df.columns.tolist()
                    objectives = ['desired_yield', 'undesired_yield']
                    group_cols = [c for c in all_cols if c not in objectives]
                
                # Extract parameter selections for this run
                grouped_list = list(df.groupby(group_cols, sort=False))
                run_selections = []
                
                for group in grouped_list:
                    group_df = group[1]
                    selection_dict = {}
                    for param_name in param_options.keys():
                        if param_name in group_df.columns:
                            all_values = group_df[param_name].tolist()
                            selection_dict[param_name] = all_values[0]
                    run_selections.append(selection_dict)
                
                # Skip first selection for Bayesian methods
                if "bayesian" in method_path:
                    run_selections = run_selections[1:]
                
                # Compute cumulative entropy for this run
                run_entropy = compute_individual_run_cumulative_entropy(run_selections, param_options)
                run_entropies.append(run_entropy)
                
            except Exception as e:
                print(f'Error processing {run_dir}: {e}')
                continue
        
        # Store entropy data for this method
        if run_entropies:
            entropy_data['chan_lam_full'][method_path]['cumulative_entropies'] = run_entropies
            entropy_data['chan_lam_full'][method_path]['median'] = float(np.median(run_entropies))
            entropy_data['chan_lam_full'][method_path]['q1'] = float(np.percentile(run_entropies, 25))
            entropy_data['chan_lam_full'][method_path]['q3'] = float(np.percentile(run_entropies, 75))
        
        print(f"Processed Chan_Lam_Full - {method_path.split('/')[-1]}: {len(run_entropies)} runs")
    
    return entropy_data

def create_performance_comparison_plot(performance_data, with_paper=True, save_path="./pngs/"):
    """Create performance comparison plots for all three aggregation methods
    
    Args:
        performance_data: Performance data dictionary
        with_paper: If True, analyze LLM with-paper; if False, analyze LLM without-paper
        save_path: Directory to save figures
    """
    
    if not with_paper:
        fig, axes = plt.subplots(2, 1, figsize=(30, 16))
        aggregation_methods = ['mean', 'max']
        aggregation_titles = ['Average (Mean)', 'Upper-Bound (Max)']
    else:    
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
        aggregation_methods = ['min', 'mean', 'max']
        aggregation_titles = ['Lower-Bound (Min)', 'Average (Mean)', 'Upper-Bound (Max)']
    paper_status = "with-paper" if with_paper else "without-paper"
    
    for idx, (agg_method, title) in enumerate(zip(aggregation_methods, aggregation_titles)):
        ax = axes[idx]
        
        # Separate LLM and BO methods based on with_paper parameter
        llm_data = {}
        bo_data = {}
        
        for method_key, method_info in performance_data[agg_method]['chan_lam_full'].items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '')
            method_name = method_name.replace('-preview-03-25', '').replace('-20250514', '')
            method_name = method_name.replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '').replace('-20250805', '')
            if 'gpt-4' in method_name or 'worst' in method_name or '4-5' in method_name or '3-5-sonnet' in method_name:
                continue
            
            # Filter based on with_paper parameter
            if 'llm' in method_key:
                has_paper = '-paper' in method_key
                if (with_paper and has_paper) or (not with_paper and not has_paper):
                    llm_data[method_name] = method_info['top_obs']
            elif 'bayesian' in method_key:
                bo_data[method_name] = method_info['top_obs']
        
        # Create boxplots
        position = 0
        all_data = []
        all_labels = []
        colors = []
        
        # Add LLM data
        for method_name, data in sorted(llm_data.items()):
            all_data.append(data)
            all_labels.append(method_name.replace('-with-paper', '').replace('-paper', '').replace('-thinking', '\nthinking').replace('-medium', '\nmedium'))
            colors.append('#ff7f0e')  # Orange for LLM
            position += 1
        
        # Add separator
        if llm_data and bo_data:
            position += 0.5
        
        # Add BO data
        for method_name, data in sorted(bo_data.items()):
            all_data.append(data)
            all_labels.append(method_name)
            colors.append('#1f77b4')  # Blue for BO
            position += 1
        
        if all_data:
            positions = list(range(len(all_data)))
            boxplots = ax.boxplot(all_data, positions=positions, patch_artist=True, 
                                showfliers=True, widths=0.6)
            
            # Color the boxplots
            for patch, color in zip(boxplots['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(boxplots[element], color='black')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(all_labels, rotation=0, ha='center', fontsize=10)
        
        ax.set_title(f'{title}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weighted Selectivity', fontsize=12)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        
        # No theoretical maximum line since it varies by aggregation
    
    # No suptitle for cleaner look
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    filename = f'figure_S14_performance_{paper_status}.png'
    plt.savefig(os.path.join(save_path, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_entropy_comparison_plot(entropy_data, with_paper=True, save_path="./pngs/"):
    """Create entropy comparison plot
    
    Args:
        entropy_data: Entropy data dictionary
        with_paper: If True, analyze LLM with-paper; if False, analyze LLM without-paper
        save_path: Directory to save figures
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    paper_status = "with-paper" if with_paper else "without-paper"
    
    # Separate LLM and BO methods based on with_paper parameter
    llm_data = {}
    bo_data = {}
    
    for method_key, method_info in entropy_data['chan_lam_full'].items():
        if not method_info['cumulative_entropies']:
            continue
            
        method_name = method_key.split('/')[-1]
        method_name = method_name.replace('-1-20-20', '').replace('-latest', '')
        method_name = method_name.replace('-preview-03-25', '').replace('-20250514', '')
        method_name = method_name.replace('-preview-04-17', '').replace('-des0', '-des')
        method_name = method_name.replace('-preview-06-17', '').replace('-20250805', '')
        
        # Filter based on with_paper parameter
        if 'llm' in method_key:
            has_paper = '-paper' in method_key
            if (with_paper and has_paper) or (not with_paper and not has_paper):
                llm_data[method_name] = method_info['cumulative_entropies']
        elif 'bayesian' in method_key:
            bo_data[method_name] = method_info['cumulative_entropies']
    
    # Create boxplots
    all_data = []
    all_labels = []
    colors = []
    
    # Add LLM data
    for method_name, data in sorted(llm_data.items()):
        all_data.append(data)
        all_labels.append(method_name.replace('-with-paper', '').replace('-paper', ''))
        colors.append('#ff7f0e')  # Orange for LLM
    
    # Add BO data
    for method_name, data in sorted(bo_data.items()):
        all_data.append(data)
        all_labels.append(method_name)
        colors.append('#1f77b4')  # Blue for BO
    
    if all_data:
        positions = list(range(len(all_data)))
        boxplots = ax.boxplot(all_data, positions=positions, patch_artist=True, 
                            showfliers=True, widths=0.6)
        
        # Color the boxplots
        for patch, color in zip(boxplots['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(boxplots[element], color='black')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(all_labels, rotation=0, ha='center', fontsize=10)
    
    # No title for cleaner look
    ax.set_ylabel('Cumulative Entropy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    filename = f'figure_S14_entropy_{paper_status}.png'
    plt.savefig(os.path.join(save_path, filename), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_statistical_comparison_matrices(performance_data, with_paper=True, save_path="./pngs/"):
    """Create matrix-style statistical comparison figures similar to S1/S2 and S6/S7
    
    Args:
        performance_data: Performance data dictionary
        with_paper: If True, analyze LLM with-paper; if False, analyze LLM without-paper
        save_path: Directory to save figures
    """
    from matplotlib.colors import ListedColormap
    
    paper_status = "with-paper" if with_paper else "without-paper"
    paper_label = "with Paper" if with_paper else "without Paper"
    
    # Create separate figures for p-values and effect sizes for each aggregation
    aggregation_methods = ['min', 'mean', 'max']
    aggregation_titles = ['Lower-Bound (Min)', 'Average (Mean)', 'Upper-Bound (Max)']
    
    for agg_idx, (agg_method, agg_title) in enumerate(zip(aggregation_methods, aggregation_titles)):
        # Separate LLM and BO methods based on with_paper parameter
        llm_data = {}
        bo_data = {}
        
        for method_key, method_info in performance_data[agg_method]['chan_lam_full'].items():
            method_name = method_key.split('/')[-1]
            method_name = method_name.replace('-1-20-20', '').replace('-latest', '')
            method_name = method_name.replace('-preview-03-25', '').replace('-20250514', '')
            method_name = method_name.replace('-preview-04-17', '').replace('-des0', '-des')
            method_name = method_name.replace('-preview-06-17', '').replace('-20250805', '')
            
            # Filter based on with_paper parameter
            if 'llm' in method_key:
                has_paper = '-paper' in method_key
                if (with_paper and has_paper) or (not with_paper and not has_paper):
                    llm_data[method_name] = method_info['top_obs']
            elif 'bayesian' in method_key:
                bo_data[method_name] = method_info['top_obs']
        
        if not llm_data or not bo_data:
            continue
            
        # Create matrices
        llm_names = sorted(llm_data.keys())
        bo_names = sorted(bo_data.keys())
        
        p_matrix = np.zeros((len(llm_names), len(bo_names)))
        effect_matrix = np.zeros((len(llm_names), len(bo_names)))
        
        for i, llm_name in enumerate(llm_names):
            for j, bo_name in enumerate(bo_names):
                llm_vals = llm_data[llm_name]
                bo_data_vals = bo_data[bo_name]
                
                p_val, effect_size, _ = perform_statistical_analysis(
                    llm_vals, bo_data_vals, f"{llm_name} ({paper_label})", bo_name
                )
                
                p_matrix[i, j] = p_val if p_val is not None else 1.0
                effect_matrix[i, j] = effect_size if effect_size is not None else 0.0
        
        # Create two separate figures for this aggregation method
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        
        # Figure 1: P-values with significance coloring
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
                text_color = 'white' if color_value in [0, 3] else 'black'
                
                ax1.text(j, i, text, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color=text_color)
        
        ax1.set_title(f'{agg_title} - Wilcoxon p-values\nLLM {paper_label} vs BO', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(bo_names)))
        ax1.set_yticks(range(len(llm_names)))
        ax1.set_xticklabels(bo_names, rotation=45, ha='right')
        ax1.set_yticklabels([name.replace('-with-paper', '').replace('-paper', '') for name in llm_names])
        ax1.set_xlabel('BO Methods', fontweight='bold')
        ax1.set_ylabel('LLM Methods', fontweight='bold')
        
        # Add colorbar for p-values
        cbar1 = fig1.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.6)
        cbar1.set_label('Significance Level', fontweight='bold')
        cbar1.set_ticks([0, 1, 2, 3])
        cbar1.set_ticklabels(['p≥0.05 (ns)', 'p<0.05 (*)', 'p<0.01 (**)', 'p<0.001 (***)'])
        
        # Figure 2: Effect sizes
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
                        fontsize=10, fontweight='bold', color=text_color)
        
        ax2.set_title(f'{agg_title} - Cliff\'s δ Effect Sizes\nLLM {paper_label} vs BO', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(bo_names)))
        ax2.set_yticks(range(len(llm_names)))
        ax2.set_xticklabels(bo_names, rotation=45, ha='right')
        ax2.set_yticklabels([name.replace('-with-paper', '').replace('-paper', '') for name in llm_names])
        ax2.set_xlabel('BO Methods', fontweight='bold')
        ax2.set_ylabel('LLM Methods', fontweight='bold')
        
        # Add colorbar for effect sizes
        cbar2 = fig2.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.6)
        cbar2.set_label('Cliff\'s δ (LLM advantage →)', fontweight='bold')
        
        # Adjust layout and save
        fig1.tight_layout()
        fig2.tight_layout()
        
        # Save figures
        os.makedirs(save_path, exist_ok=True)
        fig1.savefig(os.path.join(save_path, f'figure_S14_pvalues_{agg_method}_{paper_status}.png'), 
                    dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_path, f'figure_S14_effect_sizes_{agg_method}_{paper_status}.png'), 
                    dpi=300, bbox_inches='tight')
        
        plt.show()
    
    print(f"Statistical matrix figures saved to {save_path}")
    return True

if __name__ == "__main__":
    # Get run path from command line argument or user input
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        print(f"Using run path: {run_path}")
    else:
        run_path = input('Enter the run path: ')
    
    n_tracks = 20
    track_size = 20

    # Load path_dict for Chan Lam (both with-paper and without-paper LLM runs)
    path_dict = {}
    path_dict['Chan_Lam_Full'] = {}
    
    # BO paths
    bo_path = os.path.join(run_path, 'bayesian/Chan_Lam_Full/benchmark/')
    if os.path.exists(bo_path):
        bo_paths = [os.path.join(bo_path, path) for path in os.listdir(bo_path)]
        for path in bo_paths:
            if path.endswith('-20') or path.endswith('-20-des0'):
                if os.path.exists(path):
                    path_dict['Chan_Lam_Full'][path] = None
    
    # LLM paths - include BOTH with-paper and without-paper runs
    llm_path = os.path.join(run_path, 'llm/Chan_Lam_Full/benchmark/')
    if os.path.exists(llm_path):
        llm_paths = [os.path.join(llm_path, path) for path in os.listdir(llm_path)]
        for path in llm_paths:
            # Include all proper LLM runs (both with and without -paper tag)
            if '1-20-20' in path:
                if os.path.exists(path):
                    path_dict['Chan_Lam_Full'][path] = None

    print(f"Found {len(path_dict['Chan_Lam_Full'])} methods for Chan Lam analysis")
    for method_path in path_dict['Chan_Lam_Full'].keys():
        method_type = "BO" if 'bayesian' in method_path else "LLM"
        paper_status = "with-paper" if '-paper' in method_path else "without-paper"
        print(f"  {method_type}: {method_path.split('/')[-1]} ({paper_status})")

    # Get parameter options for Chan Lam
    dataset_param_options = {}
    param_options = {}
    dataset = Dataset('Chan_Lam_Full')
    for param in dataset.param_space:
        param_options[param.name] = param.options
    dataset_param_options['Chan_Lam_Full'] = param_options

    print("\nExtracting performance data with all aggregation methods...")
    performance_data = get_performance_data_all_aggregations(path_dict)
    
    print("\nExtracting entropy data...")
    entropy_data = get_entropy_data(path_dict, dataset_param_options)
    
    # Generate figures for LLM WITH paper
    print("\n" + "="*60)
    print("GENERATING FIGURES FOR LLM WITH PAPER")
    print("="*60)
    
    print("\nCreating performance comparison plots (with-paper)...")
    # perf_fig_with_paper = create_performance_comparison_plot(performance_data, with_paper=True)
    
    # print("\nCreating entropy comparison plot (with-paper)...")
    # entropy_fig_with_paper = create_entropy_comparison_plot(entropy_data, with_paper=True)
    
    # print("\nCreating statistical comparison matrices (with-paper)...")
    # create_statistical_comparison_matrices(performance_data, with_paper=True)
    
    # Generate figures for LLM WITHOUT paper
    print("\n" + "="*60)
    print("GENERATING FIGURES FOR LLM WITHOUT PAPER")
    print("="*60)
    
    print("\nCreating performance comparison plots (without-paper)...")
    perf_fig_without_paper = create_performance_comparison_plot(performance_data, with_paper=False)
    
    print("\nCreating entropy comparison plot (without-paper)...")
    entropy_fig_without_paper = create_entropy_comparison_plot(entropy_data, with_paper=False)
    
    print("\nCreating statistical comparison matrices (without-paper)...")
    create_statistical_comparison_matrices(performance_data, with_paper=False)
    
    print("\n" + "="*60)
    print("Figure S14 analysis complete!")
    print("="*60)
    print('\nWith-paper figures:')
    print('  Performance: ./pngs/figure_S14_performance_with-paper.png')
    print('  Entropy: ./pngs/figure_S14_entropy_with-paper.png')
    print('  P-values: ./pngs/figure_S14_pvalues_min/mean/max_with-paper.png')
    print('  Effect sizes: ./pngs/figure_S14_effect_sizes_min/mean/max_with-paper.png')
    print('\nWithout-paper figures:')
    print('  Performance: ./pngs/figure_S14_performance_without-paper.png')
    print('  Entropy: ./pngs/figure_S14_entropy_without-paper.png')
    print('  P-values: ./pngs/figure_S14_pvalues_min/mean/max_without-paper.png')
    print('  Effect sizes: ./pngs/figure_S14_effect_sizes_min/mean/max_without-paper.png')
