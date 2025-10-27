from olympus.datasets.dataset import Dataset
import numpy as np
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

dataset_names = [
    'Suzuki_Cernak',
    'amide_coupling_hte', 
    'Reductive_Amination',
    'Suzuki_Doyle',
    'Chan_Lam_Full',
    'Buchwald_Hartwig'
]
dataset_colors = {
    'buchwald_hartwig': '#000000',  # Dark blue (darkest)
    'chan_lam_full': '#0071b2',     # Dark Orange
    'suzuki_doyle': '#009e74',      # Light Blue
    'reductive_amination': '#cc797f', # Orange
    'amide_coupling_hte': '#d55e00', # Yellow  
    'suzuki_cernak': '#f0e142',      # Lighter gray
}

dataset_to_obj = {
    'Buchwald_Hartwig': 'yield',
    'Suzuki_Doyle': 'yield', 
    'Suzuki_Cernak': 'conversion',
    'Reductive_Amination': 'percent_conversion',
    'amide_coupling_hte': 'yield',
    'Chan_Lam_Full': {
        'objectives': ['desired_yield', 'undesired_yield'],
        'transform': 'weighted_selectivity',  # (desired/(desired + undesired)) * desired
        'order': [0, 1],  # desired, undesired
        'aggregation': 'min'  # Uses minimum selectivity for robustness
    }
}

def get_objective_value(obs_or_group, dataset_config, aggregation='first'):
    """
    Extract objective value from observation(s) based on dataset configuration
    
    Args:
        obs_or_group: Single observation dict or pandas group for aggregation
        dataset_config: Dataset configuration (string or dict)
        aggregation: How to aggregate multiple observations ('first', 'mean', 'max', 'min', 'median')
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
            if aggregation == 'first':
                objectives = [obs_or_group[obj].iloc[0] for obj in dataset_config['objectives']]
            elif aggregation == 'mean':
                objectives = [obs_or_group[obj].mean() for obj in dataset_config['objectives']]
            elif aggregation == 'max':
                objectives = [obs_or_group[obj].max() for obj in dataset_config['objectives']]
            elif aggregation == 'min':
                objectives = [obs_or_group[obj].min() for obj in dataset_config['objectives']]
            elif aggregation == 'median':
                objectives = [obs_or_group[obj].median() for obj in dataset_config['objectives']]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Single observation
            objectives = [obs_or_group[obj] for obj in dataset_config['objectives']]
        
        # Apply transformation
        if dataset_config['transform'] == 'subtract':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] - objectives[order[1]]
        elif dataset_config['transform'] == 'weighted_selectivity':
            order = dataset_config.get('order', [0, 1])
            desired = objectives[order[0]]
            undesired = objectives[order[1]]
            total = desired + undesired
            if total > 0:
                selectivity_ratio = desired / total
                return selectivity_ratio * desired
            else:
                return 0.0
        elif dataset_config['transform'] == 'add':
            return sum(objectives)
        elif dataset_config['transform'] == 'multiply':
            result = 1
            for obj in objectives:
                result *= obj
            return result
        elif dataset_config['transform'] == 'divide':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] / objectives[order[1]] if objectives[order[1]] != 0 else 0
        elif dataset_config['transform'] == 'ratio':
            order = dataset_config.get('order', [0, 1])
            return objectives[order[0]] / (objectives[order[0]] + objectives[order[1]]) if (objectives[order[0]] + objectives[order[1]]) != 0 else 0
        else:
            raise ValueError(f"Unknown transform: {dataset_config['transform']}")

def calculate_parameter_importance(dataset_name, dataset):
    """
    Calculate parameter importance variation using Random Forest feature importance.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object with data and param_space attributes
    
    Returns:
        importance_variation: Standard deviation of normalized parameter importances
    """
    # Get the raw data
    data = dataset.data
    
    # Extract parameter names
    param_names = [param.name for param in dataset.param_space.parameters]
    
    if dataset_name == 'Chan_Lam_Full':
        # Special handling for Chan_Lam_Full - group by parameters like in get_tracks
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Get parameter names (exclude objectives)
        exclude_keys = {'desired_yield', 'undesired_yield'}
        param_names_filtered = [col for col in df.columns if col not in exclude_keys]
        
        # Group by parameter combinations and aggregate
        param_groups = df.groupby(param_names_filtered)
        processed_data = []
        
        for name, group in param_groups:
            # Get parameter values (first row of group since they're all the same)
            param_values = group[param_names_filtered].iloc[0].tolist()
            
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
            
            # Use minimum weighted selectivity for the parameter group (worst-case/lower bound)
            if weighted_selectivities:
                obj_value = np.min(weighted_selectivities)
            else:
                obj_value = 0.0
            
            # Combine parameters and objective
            row_data = param_values + [obj_value]
            processed_data.append(row_data)
        
        # Create new dataframe with processed data
        columns = param_names_filtered + ['objective']
        processed_df = pd.DataFrame(processed_data, columns=columns)
        
        # Features (X) and target (y)
        X = processed_df.iloc[:, :-1]  # All columns except last
        y = processed_df.iloc[:, -1]   # Last column (objective)
        param_names = param_names_filtered  # Use filtered parameter names
        
    else:
        # Standard handling for other datasets
        # Features (X) are all columns except the last one (which is the objective)
        X = data.iloc[:, :len(param_names)]
        # Target (y) is the last column
        y = data.iloc[:, len(param_names):]
        if y.shape[1] > 1:
            y = y.iloc[:, 0] - y.iloc[:, 1]  # Handle multi-objective case
        else:
            y = y.iloc[:, 0]
    
    # One-hot encode categorical parameters
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=None, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    model.fit(X_encoded, y)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Map feature importances back to original parameters
    param_importances = {}
    start_idx = 0
    
    for param_name in param_names:
        # Find parameter in dataset.param_space
        param_obj = next(p for p in dataset.param_space.parameters if p.name == param_name)
        n_values = len(param_obj.options)
        # Sum importances for all one-hot features of this parameter
        param_importance = np.sum(feature_importances[start_idx:start_idx + n_values])
        param_importances[param_name] = param_importance
        start_idx += n_values
    
    # Normalize importances to sum to 1
    total_importance = sum(param_importances.values())
    normalized_importances = {k: round(float(v/total_importance), 3) for k, v in param_importances.items()}
    
    # Calculate variation in parameter importances
    importance_values = list(normalized_importances.values())
    importance_balance = 1 - round(float(np.std(importance_values)), 3)

    # Get the theoretical minimum PIB
    n = len(param_names)
    theoretical_min_pib = 1 - (np.sqrt(n - 1) / n)
    print(f"Theoretical minimum PIB: {theoretical_min_pib}, n = {n}")

    # Normalize the PIB based on the theoretical minimum
    normalized_pib = round((importance_balance - theoretical_min_pib) / (1 - theoretical_min_pib), 4)
    print(f"Normalized PIB: {normalized_pib}")

    # Get r2 score
    predictions = model.predict(X_encoded)
    from sklearn.metrics import r2_score
    r2 = round(float(r2_score(y, predictions)), 3)
    
    return importance_balance, normalized_importances, r2

def calculate_radar_area_triangles(values):
    """Calculate area as sum of triangles from center"""
    n = len(values)
    angle_step = 2 * np.pi / n
    
    area = 0
    for i in range(n):
        j = (i + 1) % n
        # Area of triangle = 0.5 * r1 * r2 * sin(angle_between)
        triangle_area = 0.5 * values[i] * values[j] * np.sin(angle_step)
        area += triangle_area
    
    return round(area, 3)

# Assuming normalized_df is already prepared
def create_plotly_radar_charts(df, title, normalize=False, use_theoretical_max=False, use_theoretical_min=False, use_min_area=False, use_max_area=False):
    # only one of use_theoretical_max, use_theoretical_min, use_min_area, use_max_area can be True
    assert sum([use_theoretical_max, use_theoretical_min, use_min_area, use_max_area]) <= 1, "Only one of use_theoretical_max, use_theoretical_min, use_min_area, use_max_area can be True"
    # Require one to be True
    assert any([use_theoretical_max, use_theoretical_min, use_min_area, use_max_area]), "At least one of use_theoretical_max, use_theoretical_min, use_min_area, use_max_area must be True"

    has_theoretical = 'theoretical' in df.index
    if has_theoretical:
        theoretical_normalized_values = []
        values = df.loc['theoretical'].tolist()
        if normalize:
            max_values = df.max()
            for v, c in zip(values, df.columns):
                theoretical_normalized_values.append(v / max_values[c])
        else:
            theoretical_normalized_values = values
        df = df.drop(index='theoretical')
    # Create subplot grid
    n_datasets = len(df.index)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    # Create specs - all polar except the last position
    specs = [[{'type': 'polar'}] * n_cols for _ in range(n_rows)]
    
    # If there's an empty subplot, make it xy for the bar chart
    total_subplots = n_rows * n_cols
    if n_datasets < total_subplots:
        specs[-1][-1] = {'type': 'xy'}  # Change last subplot to xy
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs
    )
    
    # Add radar charts only for datasets
    areas = []
    out_vals = []
    
    for i, dataset in enumerate(df.index):
        if dataset == 'Suzuki_Doyle':
            name = 'Suzuki Yield'
        elif dataset == 'Suzuki_Cernak':
            name = 'Suzuki Conversion'
        else:
            print(dataset)
            name = dataset.replace('_', ' ')
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        values = df.loc[dataset].tolist()
        if normalize:
            max_values = df.max()
            normalized_values = []
            for v, c in zip(values, df.columns):
                # print(v, c)
                normalized_values.append(v / max_values[c])
            # Add a row to the out_df for the dataset
            out_vals.append(normalized_values)
        else:
            normalized_values = values
            out_vals.append(normalized_values)
            
        area = calculate_radar_area_triangles(normalized_values)
        areas.append(area)
        print(f"{name} area: {area}")
        
        categories = df.columns.tolist()
        
        fig.add_trace(
            go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill='toself',
                name=name,
                line=dict(color=dataset_colors.get(dataset.lower(), 'blue'), shape='linear'),
                fillcolor=dataset_colors.get(dataset.lower(), 'blue'),
                opacity=0.7,
                showlegend=True,
                mode='lines',
                connectgaps=True
            ),
            row=row, col=col
        )
        
        fig.update_polars(
            radialaxis=dict(
                visible=False,
                range=[0, 1],
                tickvals=[],
                ticktext=[]
            ),
            bgcolor="gray",
            row=row, col=col
        )
    
    # Add bar chart in the last subplot only if there's an empty space
    if n_datasets < total_subplots:
        fig.add_trace(
            go.Bar(
                x=[name.replace('_', ' ') for name in df.index],
                y=[a/max(areas) for a in areas],
                marker=dict(
                    color=[dataset_colors.get(name.lower(), 'blue') for name in df.index],
                    line=dict(color='black', width=2)
                ),
                showlegend=False,
                opacity=0.7
            ),
            row=n_rows, col=n_cols
        )
        
        # Update bar chart layout
        fig.update_xaxes(
            title_text="Dataset",
            tickangle=0,
            showticklabels=False,
            row=n_rows, col=n_cols
        )
        fig.update_yaxes(
            title_text="Complexity Score", 
            row=n_rows, col=n_cols
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="SF Pro", color="black"),
            x=0.5
        ),
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=12, family="SF Pro", color="black")
        ),
        margin=dict(t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    out_df = pd.DataFrame(out_vals, index=df.index, columns=df.columns)

    if has_theoretical:
        theoretical_area = calculate_radar_area_triangles(theoretical_normalized_values)
        scores = [a/theoretical_area for a in areas]
    else:
        if use_theoretical_max:
            max_area = calculate_radar_area_triangles([1, 1, 1, 1, 1, 1])
            scores = [a/max_area for a in areas]
        elif use_theoretical_min:
            min_area = calculate_radar_area_triangles([0, 0, 0, 0, 0, 0])
            scores = [a/min_area for a in areas]
        elif use_min_area:
            min_area = min(areas)
            scores = [a/min_area for a in areas]
        elif use_max_area:
            max_area = max(areas)
            scores = [a/max_area for a in areas]
    return fig, scores, out_df

if __name__ == "__main__":
    # Note: Figure 3 uses Olympus datasets directly, no run data needed
    if len(sys.argv) > 1:
        print(f"Note: Figure 3 doesn't use run data. Ignoring provided path: {sys.argv[1]}")
    
    from scipy.stats import skew

    dataset_info = {}
    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name)
        n_params = len(dataset.param_space.parameters)
        raw_data = dataset.data
        
        if dataset_name == 'Chan_Lam_Full':
            # Special handling for Chan_Lam_Full - group by parameters like in get_tracks
            import pandas as pd
            df = pd.DataFrame(raw_data)
            
            # Get parameter names (exclude objectives)
            exclude_keys = {'desired_yield', 'undesired_yield'}
            param_names = [col for col in df.columns if col not in exclude_keys]
            
            # Group by parameter combinations
            param_groups = df.groupby(param_names)
            obj_data = []
            
            for name, group in param_groups:
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
                
                # Use minimum weighted selectivity for the parameter group (worst-case/lower bound)
                if weighted_selectivities:
                    min_weighted_selectivity = np.min(weighted_selectivities)
                    obj_data.append(min_weighted_selectivity)
                else:
                    obj_data.append(0.0)
            
            space_size = len(obj_data)  # Number of unique parameter combinations
        else:
            # Standard handling for other datasets
            obj_data = dataset.data.iloc[:, n_params:].values
            if obj_data.shape[1] == 1:
                obj_data = obj_data.flatten()
            else:
                obj_data = obj_data[:, 0] - obj_data[:, 1]
            obj_data = obj_data.tolist()
            space_size = len(raw_data)
    
        # Calculate statistics on the processed objective data
        min_obj = min(obj_data)
        max_obj = max(obj_data)
        std_obj = float(np.std(obj_data))
        med_obj = float(np.median(obj_data))
        q1_obj = float(np.percentile(obj_data, 25))
        q3_obj = float(np.percentile(obj_data, 75))
        mean_obj = float(np.mean(obj_data))
        objective_skewness = round(float(skew(obj_data, bias=True)), 3)
        elite_threshold = 0.95 * max_obj
        scarcity_index = 1 - round(float(np.mean(np.array(obj_data) >= elite_threshold)), 3)
        
        param_options = {}
        for param in dataset.param_space.parameters:
            param_options[param.name] = len(param.options)
        options = list(param_options.values())
        
        print(f"{dataset_name} has {space_size} design points")
        
        dataset_info[dataset_name] = {
            'space_size': space_size,
            'n_params': n_params,
            'n_options': options,
            'min_obj': round(min_obj, 3),
            'max_obj': round(max_obj, 3),
            'std_obj': round(std_obj, 3),
            'med_obj': round(med_obj, 3),
            'q1_obj': round(q1_obj, 3),
            'q3_obj': round(q3_obj, 3),
            'mean_obj': round(mean_obj, 3),
            'avg_num_options': round(float(np.mean(options)), 3),
            'objective_skewness': objective_skewness,
            'elite_threshold': elite_threshold,
            'scarcity_index': scarcity_index
        }

    # Calculate parameter importance variation for each dataset
    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name)
        importance_balance, param_importances, r2 = calculate_parameter_importance(dataset_name, dataset)        
        # Add to dataset_info
        dataset_info[dataset_name]['param_importance_balance'] = round(importance_balance, 4)
        dataset_info[dataset_name]['param_importances'] = {k: round(v, 4) for k, v in param_importances.items()}

    df = pd.DataFrame(dataset_info).T
    df = df[['space_size', 'n_params', 'avg_num_options', 'objective_skewness', 'scarcity_index', 'param_importance_balance']]
    df = df.reindex([
        'Suzuki_Cernak',
        'amide_coupling_hte', 
        'Reductive_Amination',
        'Suzuki_Doyle',
        'Chan_Lam_Full',
        'Buchwald_Hartwig'
    ])
    df = df.rename(columns={'space_size': 'PSS', 'n_params': 'NP', 'avg_num_options': 'AOP', 'objective_skewness': 'SKEW', 'scarcity_index': 'SI', 'param_importance_balance': 'PIB'})

    plotly_fig, scores, out_df = create_plotly_radar_charts(df, 'Comparison of Optimization Complexity', normalize=True, use_min_area=True)

    # Save the figure to ./pngs/figure_3.png
    os.makedirs('./pngs', exist_ok=True)
    plotly_fig.write_image('./pngs/figure_3.png', format='png', width=1200, height=800)

    print(f'Figure 3 saved to ./pngs/figure_3.png')
        