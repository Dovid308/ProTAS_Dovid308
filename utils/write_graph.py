import numpy as np
import os
from itertools import groupby
import tqdm
import pickle
from pathlib import Path

# ==============================================================
# PATH SETTINGS (Relative Path)
# ==============================================================
BASE_PATH = Path("data/EgoPER_action_segmentation")

def write_graph_from_transcripts(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write a task graph from transcripts of actions for a single recipe.
    Uses the ORIGINAL logic: reads from local 'groundTruth' and local 'mapping.txt'.

    Parameters:
    - dataset (str): The name of the dataset (e.g., 'coffee').
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the local mapping file.
    """
    recipe_path = BASE_PATH / dataset
    gt_path = recipe_path / 'groundTruth'
    mapping_file = recipe_path / 'mapping.txt'
    graph_path = recipe_path / 'graph'
    
    # Ensure graph directory exists (parents=True prevents FileNotFoundError)
    graph_path.mkdir(parents=True, exist_ok=True)
    
    # Create a dictionary to map actions to indices (from LOCAL mapping)
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            actions = line.strip().split(map_delimiter, 1)
            # Depending on the local mapping format, it could be "ID ACTION" or "ACTION ID"
            # Assuming "ID ACTION" based on original code structure
            if len(actions) == 2:
                actions_dict[actions[1].strip()] = int(actions[0].strip())
    
    # Initialize matrices for predecessor and successor relationships
    pre_mat = np.zeros([len(actions_dict), len(actions_dict)])
    suc_mat = np.zeros([len(actions_dict), len(actions_dict)])
    count = np.zeros([len(actions_dict)])
    
    if not gt_path.exists():
        print(f"⚠️ Missing directory: {gt_path}")
        return

    # Process each video in the local ground truth path
    for vid in tqdm.tqdm(os.listdir(gt_path), desc=f"Graph {dataset}"):
        if not vid.endswith('.txt'): continue

        with open(gt_path / vid, 'r') as f:
            content = f.read().split('\n')[:-1]
        
        classes = np.zeros([len(content)], dtype=np.int32)
        
        # Map each action in the content to its corresponding index
        for i in range(len(classes)):
            if content[i] in actions_dict:
                classes[i] = actions_dict[content[i]]
            else:
                classes[i] = 0 # Fallback to background if not found
        
        # Filter out background classes
        classes_wo_bg = [a for a in classes if a not in bg_class]
        transcript = [k for k, v in groupby(classes_wo_bg)]
        
        # Update counts and matrices
        for a in transcript:
            count[a] += 1
        for pre_action, suc_action in zip(transcript[:-1], transcript[1:]):
            pre_mat[pre_action, suc_action] += 1
            suc_mat[suc_action, pre_action] += 1
    
    # Normalize the matrices
    pre_mat = pre_mat / np.maximum(count[None, :], 1e-5)
    suc_mat = suc_mat / np.maximum(count[None, :], 1e-5)
    
    # Save the graph
    graph = {'matrix_pre': pre_mat, 'matrix_suc': suc_mat}
    with open(graph_path / 'graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    print(f"Finished writing graph for {dataset} in {graph_path}")


def write_global_graph(datasets, bg_class=[0], map_delimiter='|'):
    """
    Generate a unified graph by combining all recipes.
    This function uses 'groundTruth_global' and 'mapping_global.txt'.
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

    N = len(actions_dict)
    pre_mat = np.zeros([N, N])
    suc_mat = np.zeros([N, N])
    count = np.zeros([N])

    # 2. Iterate through all recipes in the unified ground truth
    for dataset in datasets:
        gt_path = BASE_PATH / dataset / 'groundTruth_global'
        if not gt_path.exists(): 
            continue

        for vid in tqdm.tqdm(os.listdir(gt_path), desc=f"Global Graph ({dataset})"):
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

            # Filter out background classes and compress repeated actions
            classes_wo_bg = [a for a in classes if a not in bg_class]
            transcript = [k for k, v in groupby(classes_wo_bg)]

            for a in transcript:
                count[a] += 1

            for pre_action, suc_action in zip(transcript[:-1], transcript[1:]):
                pre_mat[pre_action, suc_action] += 1
                suc_mat[suc_action, pre_action] += 1

    # 3. Global normalization
    pre_mat = pre_mat / np.maximum(count[None, :], 1e-5)
    suc_mat = suc_mat / np.maximum(count[None, :], 1e-5)

    graph = {'matrix_pre': pre_mat, 'matrix_suc': suc_mat}

    # 4. Save the global graph
    global_graph_dir = BASE_PATH / "graph"
    global_graph_dir.mkdir(parents=True, exist_ok=True)
    
    with open(global_graph_dir / "graph.pkl", 'wb') as f:
        pickle.dump(graph, f)

    print(f"\n✅ Global graph successfully created at: {global_graph_dir}/graph.pkl")


if __name__ == '__main__':


    recipes_list = ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']
    for recipe in recipes_list:
        # Note: If your local mapping files use a space instead of a pipe, change map_delimiter to ' '
        write_graph_from_transcripts(recipe, bg_class=[0], map_delimiter='|')
    write_global_graph(recipes_list)
