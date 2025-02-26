from extract_archive import extract_specific_dirs
import os
import pandas as pd

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

if not os.path.exists("./extracted_imdb"):
    extract_specific_dirs("./aclImdb_v1.tar.gz", output_dir="./extracted_imdb")

def count_files_in_dir(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

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

# Create empty lists to store data
texts = []
labels = []
splits = []

# Iterate through train and test sets
for split in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
        path = f"./extracted_imdb/aclImdb/{split}/{sentiment}"
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(1 if sentiment == "pos" else 0)
                splits.append(split)

# Create DataFrame
df = pd.DataFrame({"text": texts, "label": labels, "split": splits})

print(f"Total samples in DataFrame: {len(df)}")
df.to_csv("imdb_reviews.csv", index=False)
