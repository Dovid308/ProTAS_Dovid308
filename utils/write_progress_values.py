from os.path import exists
import numpy as np
import os
from itertools import groupby
import tqdm
from pathlib import Path

# ==============================================================
# PATH SETTINGS (Relative Path)
# ==============================================================
BASE_PATH = Path("data/EgoPER_action_segmentation")

def write_progress_values(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write progress values for each action in a single dataset.
    Uses the ORIGINAL logic: reads from local 'groundTruth' and local 'mapping.txt'.

    Parameters:
    - dataset (str): The name of the dataset (e.g., 'coffee').
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the local mapping file.
    """
    recipe_path = BASE_PATH / dataset
    gt_path = recipe_path / 'groundTruth'
    mapping_file = recipe_path / 'mapping.txt'
    progress_path = recipe_path / 'progress'
    
    # Ensure progress directory exists (parents=True prevents FileNotFoundError)
    progress_path.mkdir(parents=True, exist_ok=True)
    
    # Create a dictionary to map actions to indices
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            actions = line.strip().split(map_delimiter, 1)
            if len(actions) == 2:
                actions_dict[actions[1].strip()] = int(actions[0].strip())
    
    if not gt_path.exists():
        print(f"⚠️ Missing directory: {gt_path}")
        return

    # Process each video in the local ground truth path
    for vid in tqdm.tqdm(os.listdir(gt_path), desc=f"Progress {dataset}"):
        if not vid.endswith('.txt'): continue

        with open(gt_path / vid, 'r') as f:
            content = f.read().splitlines()
        
        classes = np.zeros([len(content)], dtype=np.int32)
        
        # Map each action in the content to its corresponding index
        for i in range(len(classes)):
            if content[i] in actions_dict:
                classes[i] = actions_dict[content[i]]
            else:
                classes[i] = 0 # Fallback to background
        
        # Initialize progress values array
        progress_values = np.zeros([len(actions_dict), len(content)])
        cur_frame = 0
        
        # Calculate progress values for each action segment
        for k, v in groupby(classes):
            segment_length = len(list(v))
            if k not in bg_class:
                cur_progress = (np.arange(segment_length) + 1) / segment_length
                progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress
            cur_frame += segment_length
        
        # Save progress values to a file
        np.save(progress_path / (vid[:-4] + '.npy'), progress_values)
    
    print(f"Finished writing progress values for {dataset} in {progress_path}")


from os.path import exists
import numpy as np
import os
from itertools import groupby
import tqdm
from pathlib import Path

# ==============================================================
# PATH SETTINGS (Relative Path)
# ==============================================================
BASE_PATH = Path("data/EgoPER_action_segmentation")

def write_progress_values(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write progress values for each action in a single dataset.
    Uses the ORIGINAL logic: reads from local 'groundTruth' and local 'mapping.txt'.

    Parameters:
    - dataset (str): The name of the dataset (e.g., 'coffee').
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the local mapping file.
    """
    recipe_path = BASE_PATH / dataset
    gt_path = recipe_path / 'groundTruth'
    mapping_file = recipe_path / 'mapping.txt'
    progress_path = recipe_path / 'progress'
    
    # Ensure progress directory exists (parents=True prevents FileNotFoundError)
    progress_path.mkdir(parents=True, exist_ok=True)
    
    # Create a dictionary to map actions to indices
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            actions = line.strip().split(map_delimiter, 1)
            if len(actions) == 2:
                actions_dict[actions[1].strip()] = int(actions[0].strip())
    
    if not gt_path.exists():
        print(f"⚠️ Missing directory: {gt_path}")
        return

    # Process each video in the local ground truth path
    for vid in tqdm.tqdm(os.listdir(gt_path), desc=f"Progress {dataset}"):
        if not vid.endswith('.txt'): continue

        with open(gt_path / vid, 'r') as f:
            content = f.read().splitlines()
        
        classes = np.zeros([len(content)], dtype=np.int32)
        
        # Map each action in the content to its corresponding index
        for i in range(len(classes)):
            if content[i] in actions_dict:
                classes[i] = actions_dict[content[i]]
            else:
                classes[i] = 0 # Fallback to background
        
        # Initialize progress values array
        progress_values = np.zeros([len(actions_dict), len(content)])
        cur_frame = 0
        
        # Calculate progress values for each action segment
        for k, v in groupby(classes):
            segment_length = len(list(v))
            if k not in bg_class:
                cur_progress = (np.arange(segment_length) + 1) / segment_length
                progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress
            cur_frame += segment_length
        
        # Save progress values to a file
        np.save(progress_path / (vid[:-4] + '.npy'), progress_values)
    
    print(f"Finished writing progress values for {dataset} in {progress_path}")


def write_global_progress(datasets, bg_class=[0], map_delimiter='|'):
    """
    Generate unified progress values combining all recipes.
    This function uses 'groundTruth_global' and 'mapping_global.txt'.
    All generated progress files are saved in a single global directory.
    """
    mapping_file = BASE_PATH / "mapping_global.txt"
    
    # 1. Load the Global Mapping
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split(map_delimiter, 1)
            if len(parts) == 2:
                actions_dict[parts[1].strip()] = int(parts[0].strip())

    N_classes = len(actions_dict)

    # 2. Iterate through all recipes in the unified ground truth
    for dataset in datasets:
        gt_path = BASE_PATH / dataset / 'groundTruth'
        
        global_progress_dir = BASE_PATH / dataset / "progress_global"
    
        global_progress_dir.mkdir(parents=True, exist_ok=True)
        if not gt_path.exists(): 
            continue

        for vid in tqdm.tqdm(os.listdir(gt_path), desc=f"Global Progress ({dataset})"):
            if not vid.endswith('.txt'): continue
            
            with open(gt_path / vid, 'r') as f:
                content = f.read().splitlines()

            # Convert action names to numeric IDs using the global mapping
            classes_list = []
            for line in content:
                if not line.strip(): continue
                if line.strip() in actions_dict:
                    classes_list.append(actions_dict[line.strip()])
                else:
                    classes_list.append(0)
            
            classes = np.array(classes_list, dtype=np.int32)
            
            # Initialize progress values array with global dimensions (N_classes)
            progress_values = np.zeros([N_classes, len(classes)])
            cur_frame = 0
            
            # Calculate progress values for each action segment
            for k, v in groupby(classes):
                segment_length = len(list(v))
                if k not in bg_class:
                    cur_progress = (np.arange(segment_length) + 1) / segment_length
                    progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress
                cur_frame += segment_length
            
            # Save progress values to the unified global directory
            np.save(BASE_PATH / dataset / "progress_global" / (vid[:-4] + '.npy'), progress_values)
            
    print(f"\n✅ Global progress values successfully created.")


if __name__ == '__main__':
    recipes_list = ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']

    print("="*60)
    print("🎯 GENERATING INDIVIDUAL PROGRESS VALUES (Original Logic)")
    print("="*60)
    for recipe in recipes_list:
        # Note: Depending on your local legacy mapping files, map_delimiter might need to be ' ' or '|'
        write_progress_values(recipe, bg_class=[0], map_delimiter='|')

    print("\n" + "="*60)
    print("🌍 GENERATING UNIFIED GLOBAL PROGRESS VALUES")
    print("="*60)
    write_global_progress(recipes_list, bg_class=[0], map_delimiter='|')


