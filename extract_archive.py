import tarfile
import os
from pathlib import Path


def extract_targz(archive_path, extract_path=None):
    """
    Extract a tar.gz file to the specified path

    Args:
        archive_path (str): Path to the tar.gz file
        extract_path (str, optional): Path to extract files to. Defaults to current directory.
    """
    try:
        if not extract_path:
            extract_path = os.getcwd()

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Successfully extracted {archive_path} to {extract_path}")
    except Exception as e:
        print(f"Error extracting {archive_path}: {str(e)}")


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
