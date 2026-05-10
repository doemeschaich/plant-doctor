"""
Splits the PlantVillage dataset into train/val/test CSV manifests.
Run from project root: python src/split_data.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path("data/raw/plantvillage dataset/color")
OUTPUT_DIR = Path("data/processed")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_image_paths(data_dir: Path) -> pd.DataFrame:
    """Walk through class folders and collect (filepath, label) pairs."""
    records = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix in IMAGE_EXTS:
                records.append({
                    "filepath": img_path.as_posix(),
                    "label": class_dir.name,
                })
    return pd.DataFrame(records)


def split_data(df: pd.DataFrame):
    """Stratified split: train/val/test with class balance preserved."""
    # First split: train+val vs. test
    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    # Second split: train vs. val (relative size matters here)
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_relative,
        stratify=trainval_df["label"],
        random_state=RANDOM_SEED,
    )
    return train_df, val_df, test_df


def main():
    print(f"Reading images from {DATA_DIR}...")
    df = collect_image_paths(DATA_DIR)
    print(f"Found {len(df)} images across {df['label'].nunique()} classes\n")

    train_df, val_df, test_df = split_data(df)

    print("Split sizes:")
    print(f"  Train: {len(train_df):>6}  ({len(train_df)/len(df):.1%})")
    print(f"  Val:   {len(val_df):>6}  ({len(val_df)/len(df):.1%})")
    print(f"  Test:  {len(test_df):>6}  ({len(test_df)/len(df):.1%})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"\nSaved CSVs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()