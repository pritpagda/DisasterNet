import pandas as pd
import os

ANNOTATIONS_DIR = "../../data/CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all"
OUTPUT_DIR = "data/processed_humanitarian"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Update splits for humanitarian task files
splits = {
    "train": "task_humanitarian_text_img_train.tsv",
    "dev": "task_humanitarian_text_img_dev.tsv",
    "test": "task_humanitarian_text_img_test.tsv"
}

# Define which labels to keep and their binary mapping:
# Assuming humanitarian labels are those except 'not_humanitarian' (0)
label_map = {
    "not_humanitarian": 0,
    "affected_individuals": 1,
    "rescue_volunteering_or_donation_effort": 1,
    "other_relevant_information": 1,
}

def process_humanitarian_file(split_name, filename):
    path = os.path.join(ANNOTATIONS_DIR, filename)
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Drop rows missing essential columns (image, tweet_text, label_text)
    df = df.dropna(subset=["image", "tweet_text", "label_text"])

    # Filter rows where label_text is in label_map keys
    df = df[df["label_text"].isin(label_map.keys())]

    # Map label_text to binary
    df["label"] = df["label_text"].map(label_map)

    # Keep relevant columns and rename them
    df = df[["image", "tweet_text", "label"]]
    df.columns = ["image_path", "text", "label"]

    # Remove leading/trailing spaces in image_path
    df["image_path"] = df["image_path"].str.strip()

    # Save processed_informative CSV
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Processed {len(df)} rows, saved to {out_path}")

if __name__ == "__main__":
    for split, file in splits.items():
        process_humanitarian_file(split, file)
