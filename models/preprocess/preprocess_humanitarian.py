import os

import pandas as pd

ANNOTATIONS_DIR = "../../data/CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all"
OUTPUT_DIR = "data/processed_humanitarian"
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {"train": "task_humanitarian_text_img_train.tsv", "dev": "task_humanitarian_text_img_dev.tsv",
          "test": "task_humanitarian_text_img_test.tsv"}

HUMANITARIAN_LABELS = ["not_humanitarian", "rescue_volunteering_or_donation_effort",
                       "infrastructure_and_utilities_damage", "affected_individuals", "injured_or_dead_people",
                       "missing_or_found_people", "other_relevant_information"]

label_to_id = {label: idx for idx, label in enumerate(HUMANITARIAN_LABELS)}


def process_humanitarian_file(split_name, filename):
    path = os.path.join(ANNOTATIONS_DIR, filename)
    df = pd.read_csv(path, sep="\t", dtype=str)

    df = df.dropna(subset=["image", "tweet_text", "label_text"])
    df = df[df["label_text"].isin(HUMANITARIAN_LABELS)]
    df["label"] = df["label_text"].map(label_to_id)
    df = df[["image", "tweet_text", "label"]]
    df.columns = ["image_path", "text", "label"]
    df["image_path"] = df["image_path"].str.strip()

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{split_name}] Processed {len(df)} rows, saved to {out_path}")


if __name__ == "__main__":
    for split, file in splits.items():
        process_humanitarian_file(split, file)
