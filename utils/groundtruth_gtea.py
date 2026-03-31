from pathlib import Path

SPLITS_DIR = Path("data/gtea/splits")
GT_DIRNAME = "groundTruth"

for split_file in sorted(SPLITS_DIR.glob("*.bundle")):
    with split_file.open("r") as f:
        vids = [line.strip() for line in f if line.strip()]

    updated = [f"{GT_DIRNAME}/{Path(v).name}" for v in vids]

    with split_file.open("w") as f:
        f.write("\n".join(updated) + "\n")

    print(f"✅ {split_file.name}: {len(updated)} video aggiornati")

print("=== TUTTO FATTO! ===")