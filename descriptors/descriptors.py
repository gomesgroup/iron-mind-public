"""
The code in this file has been extracted from the EDBO package.

https://github.com/b-shields/edbo
"""

try:
    from mordred import Calculator, descriptors
    import pandas as pd
    from rdkit import Chem
    import numpy as np

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from rdkit import Chem
    from rdkit.Chem.Draw import IPythonConsole

    from IPython import display
    from urllib.request import urlopen
except ImportError as e:
    print(f"Error: {e}")
    print(f"Please install the requirements: pip install -r descriptors/requirements.txt")

def mordred(smiles_list, name='', dropna=False):    
    """Compute chemical descriptors for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list 
        List of SMILES strings.
    name : str
        Name prepended to each descriptor name (e.g., nBase --> name_nBase).
    dropna : bool
        If true, drop columns which contain np.NaNs.
    
    Returns
    ----------
    pandas.DataFrame
        DataFrame containing overlapping Mordred descriptors for each SMILES
        string.
    """
    
    smiles_list = list(smiles_list)
    
    # Initialize descriptor calculator with all descriptors

    calc = Calculator(descriptors)
    
    output = []
    for entry in smiles_list:
        try:
            data_i = calc(Chem.MolFromSmiles(entry)).fill_missing()
        except:
            data_i = np.full(len(calc.descriptors),np.NaN)
            
        output.append(list(data_i))
        
    descriptor_names = list(calc.descriptors)
    columns = []
    for entry in descriptor_names:
        columns.append(name + '_' + str(entry))
        
    df = pd.DataFrame(data=output, columns=columns)
    df.insert(0, name + '_SMILES', smiles_list)
    
    if dropna == True:
        df = df.dropna(axis=1)
    
    return df

# -*- coding: utf-8 -*-
"""
Chemistry Utilities

"""

# Convert from chemical name or nickname to smiles

def name_to_smiles(name):
    """Convert from chemical name to SMILES string using chemical identifier resolver.
    
    Parameters
    ----------
    name : str 
        Name or nickname of compound.
    
    Returns
    ----------
    str 
        SMILES string corresponding to chemical name.
    """
    
    name = name.replace(' ', '%20')
    
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + name + '/smiles'
        smiles = urlopen(url).read().decode('utf8')
        smiles = str(smiles)
        if '</div>' in smiles:
            return 'FAILED'
        else:
            return smiles
    except:
        return 'FAILED'

# 2D SMILES visualizations

class ChemDraw:
    """Class for chemical structure visualization."""
    
    def __init__(self, SMILES_list, row_size='auto', legends=None, 
                 ipython_svg=True):
        """        
        Parameters
        ----------
        SMILES_list : list 
            List of SMILES strings to be visualized.
        row_size : 'auto', int 
            Number of structures to include per row.
        legends : None, list
            Structure legends to include below representations.
        ipython_svg :bool 
            If true print SVG in ipython console.
        
        Returns
        ----------
        None
        """
        
        self.SMILES_list = list(SMILES_list)
        self.legends = legends
        self.mols = [Chem.MolFromSmiles(s) for s in self.SMILES_list]
        
        if row_size == 'auto':
            self.molsPerRow = len(self.mols)
        else:
            self.molsPerRow = row_size  
            
        # SVGs look nicer
        IPythonConsole.ipython_useSVG = ipython_svg
        self.SVG = ipython_svg
        
    def show(self):
        """Show 2D representation of SMILES strings.
        
        Returns
        ----------
        image 
            Visualization of chemical structures.
        """
        
        img = Chem.Draw.MolsToGridImage(self.mols, 
                    molsPerRow=self.molsPerRow, 
                    subImgSize=(200, 200), 
                    legends=self.legends, 
                    highlightAtomLists=None, 
                    highlightBondLists=None, 
                    useSVG=self.SVG)
        
        display.display(img)
    
    def export(self, path):
        """Export 2D representation of SMILES strings.
        
        Parameters
        ----------
        path : 'str'
            Export a PNG image of chemical structures to path.
        
        Returns
        ----------
        None
        """
        
        img = Chem.Draw.MolsToGridImage(self.mols, 
                    molsPerRow=self.molsPerRow, 
                    subImgSize=(500, 500), 
                    legends=self.legends, 
                    highlightAtomLists=None, 
                    highlightBondLists=None, 
                    useSVG=False)
        
        img.save(path + '.png')

# -*- coding: utf-8 -*-
"""
Math utilities

"""


# Standardization

class standard:
    """
    Class for handling target standardization.
    """
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.unstandardized = np.array([])
    
    def standardize_target(self, df, target):
        """
        Standardize target vector to zero mean and unit variance.
        """
    
        if len(df) < 2:
            return df
    
        unstandard_vector = df[target].values
        self.unstandardized = unstandard_vector
        mean, std = unstandard_vector.mean(), unstandard_vector.std()
    
        # Prevent divide by zero error
        if std == 0.0:
            std = 1e-6
        
        self.mean = mean
        self.std = std
        
        standard_vector = (unstandard_vector - mean) / std
        new_df = df.copy().drop(target, axis=1)
        new_df[target] = standard_vector
    
        return new_df
    
    def unstandardize_target(self, df, target):
        """
        Retrun target data batck to unstandardized form
        via saved mean and standard deviation.
        """
        
        if len(df) < 2:
            return df
        
        if len(df) == len(self.unstandardized):
            new_df = df.copy().drop(target, axis=1)
            new_df[target] = self.unstandardized
            return new_df
        
        standard_vector = df[target].values
        
        unstandard_vector = (standard_vector * self.std) + self.mean 
        new_df = df.copy().drop(target, axis=1)
        new_df[target] = unstandard_vector
        
        return new_df
    
    def unstandardize(self, array):
        """
        Unstandardize an array of values.
        """
        
        return (array * self.std) + self.mean 
        
        
# Fit metrics

def model_performance(pred,obs):
    """
    Compute RMSE and R^2.
    """
    
    rmse = np.sqrt(np.mean((np.array(pred) - np.array(obs)) ** 2))
    r2 = metrics.r2_score(np.array(obs),np.array(pred))
    
    return rmse, r2

# PCA
    
def pca(X, n_components=1):
    """
    PCA reduction of dimensions.
    """
    
    model = PCA(n_components=n_components, copy=True)
    model.fit(X)
    ev = sum(model.explained_variance_ratio_[:n_components])
    
    print('Explained variance = ' + str(ev))
    
    columns = ['pc' + str(i + 1) for i in range(n_components)]
    pps = model.transform(X)
    
    return pd.DataFrame(data=pps, columns=columns)
    
# Data handling class

class Data:
    """
    Class or defining experiment domains and pre-processing.
    """
    
    def __init__(self, data):
        self.data = data
        self.base_data = data
    
    def reset(self):
        self.data = self.base_data
    
    def clean(self):
        self.data = drop_single_value_columns(self.data)
        self.data = drop_string_columns(self.data)
        
    def drop(self, drop_list):
        self.data = remove_features(self.data, drop_list)
    
    def standardize(self, target='yield', scaler='minmax'):
        self.data = standardize(self.data, target, scaler=scaler)
        
    def PCA(self, target='yield', n_components=1):
        pps = pca(self.data.drop(target, axis=1), n_components=n_components)
        pps[target] = self.data[target]
        self.data = pps
        
    def uncorrelated(self, target='yield', threshold=0.7, max_features=None, ranking_method='variance'):
        self.data = uncorrelated_features(self.data, 
                                          target, 
                                          threshold=threshold,
                                          max_features=max_features,
                                          ranking_method=ranking_method)
    
    def visualize(self, experiment_index_value, svg=True):
        
        columns = self.base_data.columns.values
        smi_bool = ['SMILES' in columns[i] for i in range(len(columns))]
        index = self.base_data[self.base_data.columns[smi_bool].values]
        
        SMILES_list = index.iloc[experiment_index_value].values
        cd = ChemDraw(SMILES_list, ipython_svg=svg)
        
        try:
            entry = self.base_data[self.index_headers].iloc[[experiment_index_value]]
        except:
            entry = self.base_data.iloc[[experiment_index_value]]
            
        print('\n##################################################### Experiment\n\n',
              entry,
              '\n')
        
        return cd.show()
    
    def get_experiments(self, index_values):
        try:
            entries = self.base_data[self.index_headers].iloc[index_values]
        except:
            entries = self.base_data.iloc[index_values]
            
        return entries

# Remove columns with only a single value

def drop_single_value_columns(df):
    """
    Drop datafame columns with zero variance. Return a new dataframe.
    """
    
    keep = []
    dropped = []
    for i in range(len(df.columns)):
        if len(df.iloc[:,i].drop_duplicates()) > 1:
            keep.append(df.columns.values[i])
        else:
            dropped.append(df.columns.values[i])
    
    # print(f"DEBUG: Dropped {len(dropped)} single-value columns")
    return df[keep]
    
# Remove columns with non-numeric entries
    
def drop_string_columns(df):
    """
    Drop dataframe columns with non-numeric values. Return a new dataframe.
    """
    
    keep = []
    dropped = []
    for i in range(len(df.columns)):
        unique = df.iloc[:,i].drop_duplicates()
        keepQ = True
        for j in range(len(unique)):
            if type(unique.iloc[j]) == type(''):
                keepQ = False
                break
        if keepQ: 
            keep.append(df.columns.values[i])
        else:
            dropped.append(df.columns.values[i])
    
    # print(f"DEBUG: Dropped {len(dropped)} string columns")        
    return df[keep]
        
# Remove unwanted descriptors

def remove_features(df, drop_list):
    """
    Remove features from dataframe with columns containing substrings in
    drop_list. Return a new dataframe.
    """

    keep = []
    for column_name in list(df.columns.values):
        keepQ = True
        for substring in list(drop_list):
            if substring in column_name:
                keepQ = False
                break
        if keepQ: keep.append(column_name)
    
    return df[keep]

# Standardize
    
def standardize(df, target, scaler='standard'):
    """
    Standardize descriptors but keep target.
    """
    
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    out = scaler.fit_transform(data)
    
    new_df = pd.DataFrame(data=out, columns=data.columns)
    
    if target != None:
        new_df[target] = df[target]
    
    return new_df

# Select uncorrelated set of features
    
def uncorrelated_features(df, target, threshold=0.95, max_features=None, ranking_method='variance'):
    """
    Returns an uncorrelated set of features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with features
    target : str or None
        Target column name to exclude from correlation analysis
    threshold : float or str
        Correlation threshold for considering features as correlated (default: 0.95)
        If 'skip_correlation', skip correlation filtering and use variance-based selection
    max_features : int or None
        Maximum number of features to select. If None, select all uncorrelated features.
    ranking_method : str
        Method to rank features when max_features is specified.
        Options: 'variance', 'target_corr' (requires target), 'mean_abs_corr'
        - 'variance': Rank by feature variance (higher variance = more informative)
        - 'target_corr': Rank by absolute correlation with target (requires target)
        - 'mean_abs_corr': Rank by mean absolute correlation with all other features (lower = more unique)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with selected uncorrelated features
    """
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    # print(f"DEBUG: Starting with {data.shape[1]} features")
    # print(f"DEBUG: Using threshold = {threshold}")
    
    # Special handling for very small datasets
    if threshold == 'skip_correlation':
        # print("DEBUG: Using variance-based selection (pre-standardization)")
        
        # Rank features by variance
        feature_scores = data.var().sort_values(ascending=False)
        # print(f"DEBUG: Top 5 feature variances: {feature_scores.head()}")
        
        # Select top features by variance
        if max_features is not None:
            n_select = min(max_features, len(feature_scores))
        else:
            # For small datasets, select a reasonable number based on sample size
            n_samples = data.shape[0]
            n_select = min(max(n_samples * 3, 5), len(feature_scores))  # 3x samples or 5, whichever is larger
        
        top_features = feature_scores.head(n_select).index.tolist()
        data = data[top_features]
        # print(f"DEBUG: Selected {len(top_features)} features based on variance")
        print(f"DEBUG: Selected {len(top_features)} features based on variance:")
        for i, feature in enumerate(top_features[:5]):  # Show first 5 feature names
            variance = feature_scores[feature]
            print(f"  {i+1}. {feature}: variance = {variance:.2f}")
        if len(top_features) > 5:
            print(f"  ... and {len(top_features)-5} more features")
        
        if target != None:
            data[target] = list(df[target])
        
        return data
    
    # Regular correlation-based filtering for larger datasets
    # If max_features is specified, rank features first
    if max_features is not None:
        if ranking_method == 'variance':
            # Rank by variance (higher variance = more informative)
            feature_scores = data.var().sort_values(ascending=False)
        elif ranking_method == 'target_corr' and target is not None:
            # Rank by absolute correlation with target
            target_corr = data.corrwith(df[target]).abs().sort_values(ascending=False)
            feature_scores = target_corr
        elif ranking_method == 'mean_abs_corr':
            # Rank by mean absolute correlation with other features (lower = more unique)
            corr_temp = data.corr().abs()
            # For each feature, calculate mean correlation with all other features
            mean_corr = {}
            for col in corr_temp.columns:
                # Exclude self-correlation (which is 1.0)
                other_corrs = corr_temp[col].drop(col)
                mean_corr[col] = other_corrs.mean()
            feature_scores = pd.Series(mean_corr).sort_values(ascending=True)  # Lower is better
        else:
            # Default to variance if target_corr requested but no target available
            feature_scores = data.var().sort_values(ascending=False)
        
        # print(f"DEBUG: Top 5 feature scores: {feature_scores.head()}")
        
        # Reorder columns by ranking
        ranked_features = feature_scores.index.tolist()
        data = data[ranked_features]
    
    corr = data.corr().abs()
    # print(f"DEBUG: Correlation matrix shape: {corr.shape}")
    # print(f"DEBUG: Sample of correlation values (first feature): {corr.iloc[0, 1:6].values}")
    
    keep = []
    
    for i in range(len(corr.iloc[:,0])):
        # If we've reached max_features, stop
        if max_features is not None and len(keep) >= max_features:
            # print(f"DEBUG: Reached max_features limit of {max_features}")
            break
            
        current_feature = corr.columns[i]
        above = corr.iloc[:i,i]
        if len(keep) > 0: 
            above = above[keep]
        
        high_corr_count = len(above[above >= threshold])
        total_comparisons = len(above)
        
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])
            # print(f"DEBUG: Kept feature {i} ({current_feature[:20]}...): {high_corr_count}/{total_comparisons} high correlations")
        # else:
        #     # print(f"DEBUG: Rejected feature {i} ({current_feature[:20]}...): {high_corr_count}/{total_comparisons} high correlations >= {threshold}")
        #     if total_comparisons <= 3:  # Show correlations for small datasets
        #         # print(f"DEBUG: Correlations with kept features: {above.values}")
    
    # print(f"DEBUG: Final kept features: {len(keep)}")
    data = data[keep]
    
    if target != None:
        data[target] = list(df[target])
    
    return data

def calculate_dynamic_threshold(n_samples, min_threshold=0.3, max_threshold=0.95):
    """
    Calculate correlation threshold based on number of samples.
    
    Parameters
    ----------
    n_samples : int
        Number of samples (options)
    min_threshold : float
        Minimum threshold for very small sample sizes
    max_threshold : float
        Maximum threshold for large sample sizes
        
    Returns
    -------
    float
        Appropriate correlation threshold
    """
    if n_samples <= 5:
        # For very small datasets, correlations are unreliable
        # Return a special flag to skip correlation filtering
        return 'skip_correlation'  
    elif n_samples <= 10:
        # Moderate threshold for medium datasets
        return 0.7 + (max_threshold - 0.7) * (n_samples - 5) / 5
    else:
        # Full threshold for large datasets
        return max_threshold

def analyze_descriptor_scales(data, param_name):
    """
    Analyze the scales of selected descriptors and provide insights
    """
    print(f"\n📊 Descriptor Scale Analysis for {param_name}:")
    print("=" * 60)
    
    # Calculate statistics for each descriptor
    stats = []
    for col in data.columns:
        col_data = data[col]
        stats.append({
            'name': col,
            'min': col_data.min(),
            'max': col_data.max(), 
            'mean': col_data.mean(),
            'std': col_data.std(),
            'range': col_data.max() - col_data.min()
        })
    
    # Sort by range (largest variation first)
    stats.sort(key=lambda x: x['range'], reverse=True)
    
    print("Top 5 descriptors by variation:")
    for i, stat in enumerate(stats[:5]):
        name_short = stat['name'][:40] + "..." if len(stat['name']) > 40 else stat['name']
        print(f"  {i+1}. {name_short}")
        print(f"     Range: {stat['min']:.1f} to {stat['max']:.1f} (span: {stat['range']:.1f})")
        print(f"     Mean: {stat['mean']:.1f}, Std: {stat['std']:.1f}")
    
    # Check for very large values
    max_value = max(stat['max'] for stat in stats)
    min_value = min(stat['min'] for stat in stats) 
    
    print(f"\n🔍 Scale Assessment:")
    print(f"   Largest value: {max_value:.1f}")
    print(f"   Smallest value: {min_value:.1f}")
    
    if max_value > 1000:
        print("   ⚠️  Very large values detected (>1000)")
        print("   💡 These might be:")
        print("      - Molecular weight-related descriptors")
        print("      - Surface area descriptors")  
        print("      - Complex topological indices")
        print("      - Consider log transformation or robust scaling")
    elif max_value > 100:
        print("   ℹ️  Moderate values (100-1000) - typical for some molecular descriptors")
    else:
        print("   ✅ Normal scale values (<100)")
    
    print("=" * 60)

def get_descriptors_for_param(param, decorrelation_threshold=None, max_features=None, ranking_method='variance', standardize_for_gpr=True):
    param_options = param.options
    des = mordred(param_options, name=param.name, dropna=True)
    des = Data(des)
    print(f'Original descriptors shape: {des.data.shape}')
    des.clean()
    print(f'After cleaning: {des.data.shape}')
    
    # Calculate dynamic threshold if not provided
    if decorrelation_threshold is None:
        n_samples = len(param_options)
        decorrelation_threshold = calculate_dynamic_threshold(n_samples)
        if decorrelation_threshold == 'skip_correlation':
            print(f'Skipping correlation filtering for {n_samples} samples - using variance-based selection')
        else:
            print(f'Using dynamic threshold {decorrelation_threshold:.3f} for {n_samples} samples')
    
    # For small datasets, do feature selection BEFORE standardization
    # to preserve meaningful variance differences
    if decorrelation_threshold == 'skip_correlation':
        print("DEBUG: Doing feature selection before standardization for small dataset")
        des.uncorrelated(threshold=decorrelation_threshold, target=None, 
                         max_features=max_features, ranking_method=ranking_method)
        print(f'After decorrelation: {des.data.shape}')
        
        # Analyze descriptor scales
        analyze_descriptor_scales(des.data, param.name)
        
        # For very small datasets, we have options for standardization
        n_samples = len(param_options)
        if n_samples <= 5:
            if standardize_for_gpr:
                print(f"DEBUG: Applying robust standardization for GPR compatibility")
                # Use robust standardization for GPR
                des.standardize(target=None, scaler='standard')
                print(f'After standardization (StandardScaler): {des.data.shape}')
            else:
                print(f"DEBUG: Skipping standardization for {n_samples} samples to preserve natural scale differences")
                print(f'Final shape (no standardization): {des.data.shape}')
        else:
            # Use StandardScaler instead of MinMaxScaler for better behavior with small datasets
            des.standardize(target=None, scaler='standard')
            print(f'After standardization (StandardScaler): {des.data.shape}')
    else:
        # For larger datasets, standardize first (original approach)
        des.standardize(target=None, scaler='minmax')
        print(f'After standardization: {des.data.shape}')
        des.uncorrelated(threshold=decorrelation_threshold, target=None, 
                         max_features=max_features, ranking_method=ranking_method)
        print(f'After decorrelation: {des.data.shape}')
    
    # Set index column to option names
    des.data.index = param.options
    return des.data