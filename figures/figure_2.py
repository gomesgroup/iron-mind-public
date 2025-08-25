from olympus.datasets.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


dataset_names = ['Buchwald_Hartwig', 'Suzuki_Doyle', 'Suzuki_Cernak', 'Reductive_Amination', 'Alkylation_Deprotection', 'Chan_Lam_Full']

def gather_data():
    box_plot_data = {
        
    }
    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name)
        print(dataset_name)
        for p in dataset.param_space:
            print('  ' + p.name)
        
        if dataset_name == 'Chan_Lam_Full': # apply a special handling for the multi-objective nature, desired - undesired
            # Special handling for Chan_Lam_Full with multiple objectives
            print('  Objectives: desired_yield, undesired_yield')
            print('-'*100)
            
            raw_data = dataset.data
            # Get parameter names
            param_names = [p.name for p in dataset.param_space]
            
            # Group by parameters and calculate desired - undesired yield
            param_groups = raw_data.groupby(param_names)
            obj_data = []
            
            for name, group in param_groups:
                # Average desired and undesired yields within each parameter group (consistent with optimization analysis)
                avg_desired_yield = group['desired_yield'].mean()
                avg_undesired_yield = group['undesired_yield'].mean()
                net_selectivity = avg_desired_yield - avg_undesired_yield
                obj_data.append(net_selectivity)
            
            # Create dataset label
            dataset_label = dataset_name.replace('_', '\n').upper() + '\n(yield_difference)'
            box_plot_data[dataset_label] = obj_data
        else:
            obj = dataset.value_space[0].name
            print('-'*100)
            raw_data = dataset.data
            obj_data = dataset.data.iloc[:, -1].tolist()

            if 'cernak' in dataset_name.lower():
                dataset_name = 'Suzuki_Cernak'
            elif 'doyle' in dataset_name.lower():
                dataset_name = 'Suzuki_Doyle'
            box_plot_data[dataset_name.replace('_', '\n').upper() + f'\n({obj})'] = obj_data

    return box_plot_data

if __name__ == "__main__":
    # Note: Figure 2 uses Olympus datasets directly, no run data needed
    if len(sys.argv) > 1:
        print(f"Note: Figure 2 doesn't use run data. Ignoring provided path: {sys.argv[1]}")
    
    box_plot_data = gather_data()

    # Use consistent color scheme from other figures
    dataset_to_color = {
        'reductive_amination': '#221150',
        'buchwald_hartwig': '#5e177f',
        'chan_lam_full': '#972c7f',
        'suzuki_cernak': '#d3426d',
        'suzuki_doyle': '#f8755c',
        'alkylation_deprotection': '#febb80'
    }

    # Create figure with 3 rows, 2 columns instead of 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Create the combined boxplot in the first position
    ax_combined = axes[0, 0]

    # Map datasets to specific positions in the new 3x2 layout
    positions = [
        (0, 0, 'Alkylation_Deprotection'),
        (0, 1, "Suzuki_Doyle"),    
        (0, 2, "Suzuki_Cernak"),
        (1, 0, "Chan_Lam_Full"),
        (1, 1, "Buchwald_Hartwig"),              
        (1, 2, "Reductive_Amination")         
    ]

    # Manually assign each dataset to the correct position
    for i, pos in enumerate(positions):
        row, col, dataset_id = pos
        
        # Find the dataset that matches this ID
        dataset_name = None
        values = None
        for box_dataset_name, box_values in box_plot_data.items():
            print(dataset_id.lower(), box_dataset_name.replace('\n', '_').lower())
            if dataset_id.lower() in box_dataset_name.replace('\n', '_').lower():
                dataset_name = box_dataset_name
                values = box_values
                break
        
        if dataset_name is None or values is None:
            continue  # Skip if not found
        
        # Get color from consistent color scheme
        dataset_key = '_'.join(dataset_name.split('\n')[:-1]).lower()
        color = dataset_to_color.get(dataset_key, '#1f77b4')  # Default blue if not found
        
        # Create histogram
        ax_hist = axes[row, col]
        counts, bin_edges = np.histogram(values, bins=20)
        percentages = (counts / len(values)) * 100
        ax_hist.bar(
            bin_edges[:-1],
            percentages,
            width=np.diff(bin_edges),
            color=color,
            edgecolor='k',
            linewidth=2,
            align='edge',
            alpha=0.8
        )
        
        # Set x-axis range to show 100 tick
        x_min, x_max = min(values), max(values)
        x_range = x_max - x_min
        ax_hist.set_xlim(x_min - 0.05 * x_range, max(x_max + 0.05 * x_range, 100))
        
        # Add median line vertically
        median_val = round(np.median(values), 1)
        # ax_hist.axvline(median_val, color='cyan', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add text box for median value
        ax_hist.text(0.85, 0.95, f'Median = {median_val:.1f}',
                    transform=ax_hist.transAxes, fontsize=14,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='k'))
        
        ax_hist.set_axisbelow(True)
        ax_hist.grid(False)

    # Set font size for all x ticks
    for ax in axes.flatten():
        ax.tick_params(axis='x', labelsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)  # Make room for suptitle

    # Save the figure to ./pngs/figure_2.png
    os.makedirs('./pngs', exist_ok=True)
    plt.savefig('./pngs/figure_2.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Figure 2 saved to ./pngs/figure_2.png')