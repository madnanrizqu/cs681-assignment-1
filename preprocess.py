from extract_archive import extract_specific_dirs
import os
import pandas as pd
from pathlib import Path
import tarfile

def extract_specific_dirs(
    archive_path, output_dir="./extracted_data", target_dirs=None
):
    if target_dirs is None:
        target_dirs = ["train/pos", "train/neg", "test/pos", "test/neg"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting files to: {output_path.absolute()}")

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            for target in target_dirs:
                if f"aclImdb/{target}" in member.name:
                    tar.extract(member, output_path)

    print("Extraction completed!")

def count_files_in_dir(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

def process_imdb_data():
    if not os.path.exists("./extracted_imdb"):
        extract_specific_dirs("./aclImdb_v1.tar.gz", output_dir="./extracted_imdb")

    print(
        "Num of positive in test: ",
        count_files_in_dir("./extracted_imdb/aclImdb/test/pos"),
    )
    print(
        "Num of negative in test: ",
        count_files_in_dir("./extracted_imdb/aclImdb/test/neg"),
    )
    print(
        "Num of positive in train: ",
        count_files_in_dir("./extracted_imdb/aclImdb/train/pos"),
    )
    print(
        "Num of negative in train: ",
        count_files_in_dir("./extracted_imdb/aclImdb/train/neg"),
    )

    texts = []
    labels = []
    splits = []

    for split in ["train", "test"]:
        for sentiment in ["pos", "neg"]:
            path = f"./extracted_imdb/aclImdb/{split}/{sentiment}"
            for file in os.listdir(path):
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(1 if sentiment == "pos" else 0)
                    splits.append(split)

    df = pd.DataFrame({"text": texts, "label": labels, "split": splits})
    print(f"Total samples in DataFrame: {len(df)}")
    df.to_csv("imdb_reviews.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists("imdb_reviews.csv"):
        process_imdb_data()
    else:
        print("imdb_reviews.csv already exists. Skipping processing.")
        print("""Steps in process_imdb_data:
1. Extracts positive and negative reviews from train and test directories from the tart file
    
2. Count and print files in each directory:
    - test/pos
    - test/neg
    - train/pos
    - train/neg

3. Process text data:
    - Reads each review file from all directories
    - Stores text content, sentiment labels (1 for pos, 0 for neg)
    - Records split type (train or test)

4. Create DataFrame:
    - Combines texts, labels, and splits into pandas DataFrame
    - Saves DataFrame to 'imdb_reviews.csv'    
""")
