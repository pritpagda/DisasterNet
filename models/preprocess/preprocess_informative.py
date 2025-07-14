import pandas as pd
import os

ANNOTATIONS_DIR = "../../data/CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {
    "train": "task_informative_text_img_train.tsv",
    "dev": "task_informative_text_img_dev.tsv",
    "test": "task_informative_text_img_test.tsv"
}


def process_file(split_name, filename):
    path = os.path.join(ANNOTATIONS_DIR, filename)
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Drop rows with missing essential data
    df = df.dropna(subset=["image", "tweet_text", "label"])

    # Keep only informative or not_informative labels
    df = df[df["label"].isin(["informative", "not_informative"])]

    # Map labels to binary
    df["label"] = df["label"].map({"informative": 1, "not_informative": 0})

    # Keep only relevant columns, rename
    df = df[["image", "tweet_text", "label"]]
    df.columns = ["image_path", "text", "label"]

    # Remove any leading/trailing spaces in image_path
    df["image_path"] = df["image_path"].str.strip()

    # Save processed CSV
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Processed {len(df)} rows, saved to {out_path}")


if __name__ == "__main__":
    for split, file in splits.items():
        process_file(split, file)
