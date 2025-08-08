from olympus import Dataset
import sys
import os

# Add current directory to path to import descriptors.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import directly from descriptors.py in the same directory
from descriptors import get_descriptors_for_param
import json
import time
from sklearn.preprocessing import MinMaxScaler

def update_config_files(descriptors_dict, names_dict):
    """
    Update config.json files with new descriptors
    
    Parameters
    ----------
    descriptors_dict : dict
        Dictionary with dataset -> parameter -> descriptors
    names_dict : dict  
        Dictionary with dataset -> parameter -> descriptor_names
    """
    
    print("\nUpdating config files...")
    print("=" * 80)
    
    for dataset_name in descriptors_dict.keys():
        config_path = f"olympus_datasets/dataset_{dataset_name}/config.json"
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            continue
            
        # Load existing config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update descriptors for each categorical parameter
        updated_params = 0
        for param in config["parameters"]:
            if param["type"] == "categorical" and param["name"] in descriptors_dict[dataset_name]:
                old_desc_count = len(param["descriptors"][0]) if param["descriptors"] else 0
                
                # Update with new descriptors
                param["descriptors"] = descriptors_dict[dataset_name][param["name"]]
                
                new_desc_count = len(param["descriptors"][0])
                print(f"Updated {dataset_name}.{param['name']}: {old_desc_count} -> {new_desc_count} descriptors")
                updated_params += 1
        
        # Save updated config
        if updated_params > 0:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"✓ Updated {config_path} ({updated_params} parameters)")
        else:
            print(f"⚠ No parameters updated in {config_path}")
        
        print("-" * 60)

def update_datasets(dataset_names, use_smiles_mappings, decorrelation_threshold=None, max_features=10, ranking_method='variance', update_configs=True, standardize_for_gpr=True):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    descriptors_dict = {}
    names_dict = {}
    for dataset_name, use_smiles_mapping in zip(dataset_names, use_smiles_mappings):
        descriptors_dict[dataset_name] = {}
        names_dict[dataset_name] = {}
        dataset = Dataset(dataset_name)
        total_descs = 0
        if use_smiles_mapping:
            smiles_mapping = json.load(open(f"smiles_mappings/{dataset_name}.json"))
        else:
            smiles_mapping = None
        for param in dataset.param_space:
            if param.type == "categorical":
                original_options = param.options

                if "NO_BASE" in original_options:
                    idx = original_options.index("NO_BASE")
                    original_options[idx] = ""
                
                if use_smiles_mapping:
                    param_options = [smiles_mapping[param.name][option] for option in original_options]
                    param.options = param_options
                
                # get descriptors (mordred)
                descriptors = get_descriptors_for_param(
                    param,
                    decorrelation_threshold=decorrelation_threshold,
                    max_features=max_features, 
                    ranking_method=ranking_method,
                    standardize_for_gpr=standardize_for_gpr
                )
                descriptor_names = descriptors.columns.tolist()
                names_dict[dataset_name][param.name] = descriptor_names

                descriptors_dict[dataset_name][param.name] = [[round(x, 2) for x in row] for row in descriptors.values.tolist()]
                n_descriptors = len(descriptors_dict[dataset_name][param.name][0])
                print(f'Parameter {param.name} from {dataset_name} has {len(param.options)} options and {n_descriptors} descriptors after decorrelation')
                print('-'*100)
                total_descs += n_descriptors
            else:
                total_descs +=1
        print(f'   ({dataset_name}) Total number of descriptors: {total_descs}')
        print('-'*100)
    
    # Update config files if requested
    if update_configs:
        update_config_files(descriptors_dict, names_dict)
        print("\n✅ All config files updated successfully!")
    
    return descriptors_dict, names_dict
            
            
if __name__ == "__main__":
    dataset_names = [
        "Buchwald_Hartwig",
        "Alkylation_Deprotection",
        "Suzuki_Doyle",
        "Suzuki_Cernak",
        "Chan_Lam_Full",
        "Reductive_Amination"
    ]
    use_smiles_mappings = [
        True,
        True,
        False,
        True,
        False,
        True
    ]
    decorrelation_threshold = None  # Use dynamic thresholds
    max_features = 10
    ranking_method = 'variance'
    update_configs = True  # Set to False if you only want to compute descriptors without updating config files
    standardize_for_gpr = True  # Set to False to preserve raw scales (not recommended for GPR)
    
    print("Computing new descriptors and updating config files...")
    print("=" * 80)
    
    descriptors_dict, names_dict = update_datasets(
        dataset_names, 
        use_smiles_mappings, 
        decorrelation_threshold, 
        max_features, 
        ranking_method, 
        update_configs,
        standardize_for_gpr
    )
    
    # Create backups of computed descriptors
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("computed_descriptors", exist_ok=True)
    
    backup_desc_path = f"computed_descriptors/descriptors_dict_{timestamp}.json"
    backup_names_path = f"computed_descriptors/names_dict_{timestamp}.json"
    
    with open(backup_desc_path, "w") as f:
        json.dump(descriptors_dict, f, indent=4)
    with open(backup_names_path, "w") as f:
        json.dump(names_dict, f, indent=4)
    
    print(f"\n💾 Backup saved: {backup_desc_path}")
    print(f"💾 Backup saved: {backup_names_path}")
    print("\n🎉 All done!")
