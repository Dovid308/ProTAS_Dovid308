#!/usr/bin/python3

from pathlib import Path
import torch
import numpy as np
import random
from utils.utils_paths import resolve_entry_paths

class BatchGenerator:
    """
    

    Uso:
        # Modalità locale (singola ricetta)
        gen = BatchGenerator("...EgoPER_action_segmentation/coffee", num_classes, actions_dict, sample_rate)

        # Modalità unified (tutte le ricette)
        gen = BatchGenerator("...EgoPER_action_segmentation", num_classes, actions_dict, sample_rate)

        gen.read_data("train.split1")   # oppure "test.split1", ecc.
    """

    def __init__(self, dataset_root, num_classes, actions_dict, sample_rate, feature_transpose=False, is_unified=False):
        self.root = Path(dataset_root)
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.sample_rate = sample_rate
        self.feature_transpose = feature_transpose

        self.list_of_examples: list[str] = []


        self.index = 0

        # Auto-detection: global_splits esiste solo nella root unificata
        self.is_unified = is_unified
        print(f"[BatchGenerator] mode={'UNIFIED' if self.is_unified else 'LOCAL'} | root={self.root}")

    # ------------------------------------------------------------------
    # Iterazione
    # ------------------------------------------------------------------

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self) -> bool:
        return self.index < len(self.list_of_examples)

    # ------------------------------------------------------------------
    # Lettura split
    # ------------------------------------------------------------------

    def read_data(self, split: str = "train.split1"):
        """
        split: nome dello split (es. "train.split1")
        """
        
        filename = split if split.endswith(".bundle") else f"{split}.bundle"
        splits_dir = "splits"
        bundle_path = self.root / splits_dir / filename

        self.list_of_examples = bundle_path.read_text().strip().splitlines()
        random.shuffle(self.list_of_examples)
        print(f"[BatchGenerator] loaded {len(self.list_of_examples)} examples from {bundle_path}")

  

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def next_batch(self, batch_size: int):
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input, batch_target, batch_progress = [], [], []

        for entry in batch:
            gt_path, features_path, progress_path = resolve_entry_paths(self.root, entry, self.is_unified)

            features        = np.load(features_path)
            progress_values = np.load(progress_path)

            if self.feature_transpose:
                features = features.T

            content = gt_path.read_text().strip().splitlines()
            T_eff   = min(features.shape[1], len(content))
            classes = np.array([self.actions_dict[content[i]] for i in range(T_eff)], dtype=np.float32)

            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
            batch_progress.append(progress_values[:, ::self.sample_rate])

        # --- Padding & tensori ---
        lengths = [len(t) for t in batch_target]
        B, T    = len(batch), max(lengths)
        F       = batch_input[0].shape[0]
        P       = batch_progress[0].shape[0]

        batch_input_tensor    = torch.zeros(B, F, T,               dtype=torch.float)
        batch_target_tensor   = torch.full ((B, T),      -100,     dtype=torch.long)
        batch_progress_tensor = torch.zeros(B, P, T,               dtype=torch.float)
        mask                  = torch.zeros(B, self.num_classes, T, dtype=torch.float)

        for i in range(B):
            t = lengths[i]
            f = batch_input[i].shape[1]
            batch_input_tensor   [i, :,  :f] = torch.from_numpy(batch_input[i])
            batch_target_tensor  [i,     :t] = torch.from_numpy(batch_target[i])
            batch_progress_tensor[i, :,  :f] = torch.from_numpy(batch_progress[i])
            mask                 [i, :,  :t] = 1.0

        return batch_input_tensor, batch_target_tensor, batch_progress_tensor, mask