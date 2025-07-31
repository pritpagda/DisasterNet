import os
import pandas as pd

ANNOTATIONS_DIR = "../../data/CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all"
OUTPUT_DIR = "data/processed_damage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {
    "train": "task_damage_text_img_train.tsv",
    "dev": "task_damage_text_img_dev.tsv",
    "test": "task_damage_text_img_test.tsv"
}

DAMAGE_LABELS = [
    "severe_damage",
    "mild_damage",
    "little_or_no_damage",
]

label_to_id = {label: idx for idx, label in enumerate(DAMAGE_LABELS)}

def process_damage_file(split_name, filename):
    path = os.path.join(ANNOTATIONS_DIR, filename)
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Drop rows missing key columns
    df = df.dropna(subset=["image", "tweet_text", "label"])

    # Filter rows where label is in HUMANITARIAN_LABELS
    df = df[df["label"].isin(DAMAGE_LABELS)]

    # Map label text to numeric label
    df["label_id"] = df["label"].map(label_to_id)

    # Select and rename columns
    df = df[["image", "tweet_text", "label_id"]]
    df.columns = ["image_path", "text", "label"]

    # Strip whitespace from image_path
    df["image_path"] = df["image_path"].str.strip()

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Processed {len(df)} rows, saved to {out_path}")

if __name__ == "__main__":
    for split, file in splits.items():
        process_damage_file(split, file)
