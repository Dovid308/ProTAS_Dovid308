from pathlib import Path

def resolve_entry_paths(dataset_root, entry, is_unified=False):
    root = Path(dataset_root)
    entry_path = Path(entry)
    vid_stem = entry_path.stem

    if is_unified:
        recipe = entry_path.parts[0]
        recipe_root = root / recipe
        gt_path = root / entry
        features_path = recipe_root / "features" / f"{vid_stem}.npy"
        progress_path = recipe_root / "progress_global" / f"{vid_stem}.npy"
    else:
        gt_path = root / entry
        features_path = root / "features" / f"{vid_stem}.npy"
        progress_path = root / "progress" / f"{vid_stem}.npy"

    return gt_path, features_path, progress_path