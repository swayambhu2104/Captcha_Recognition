import zipfile
import os
import random
import argparse
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

def download_extract_data(zip_path, target_dir):
    """
    Download and extract the data from a zip file.

    Parameters:
    - zip_path (str): Path to the zip file.
    - target_dir (str): Directory to extract the data.

    Returns:
    None
    """
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

def create_directories(base_dir, sets):
    """
    Create directories for each set (train, validation, test) within the base directory.

    Parameters:
    - base_dir (str): Base directory path.
    - sets (list): List of set names.

    Returns:
    - Dictionary: Paths to directories for each set.
    """
    set_dirs = {}
    for set_name in sets:
        set_path = os.path.join(base_dir, set_name)
        os.makedirs(set_path, exist_ok=True)
        set_dirs[set_name] = set_path
    return set_dirs

def split_and_copy_data(images, labels, set_dirs, test_size=0.2, shuffle=True):
    """
    Split data into training, validation, and test sets and copy images to corresponding directories.

    Parameters:
    - images (list): List of image file paths.
    - labels (list): List of corresponding labels.
    - set_dirs (dict): Paths to directories for each set.
    - test_size (float): Percentage of data for testing.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    None
    """
    x_train, x_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(1 - test_size), shuffle=shuffle, random_state=42
    )
    
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=(test_size / (1 - test_size)), shuffle=shuffle, random_state=42
    )

    # Copy images to respective directories
    for set_name, set_images, set_labels in zip(
        ["train", "validation", "test"],
        [x_train, x_valid, x_test],
        [y_train, y_valid, y_test],
    ):
        for img_path, label in zip(set_images, set_labels):
            set_path = set_dirs[set_name]
            target_path = os.path.join(set_path, label)
            os.makedirs(target_path, exist_ok=True)
            shutil.copy(img_path, target_path)

def view_random_images(target_dir, num_images=4):
    """
    View random images from the target directory.

    Parameters:
    - target_dir (str): Directory containing images.
    - num_images (int): Number of random images to display.

    Returns:
    None
    """
    target_folder = Path(target_dir)
    image_paths = list(target_folder.glob("*.jpg"))
    random_image_paths = random.sample(image_paths, num_images)

    _, ax = plt.subplots(1, num_images, figsize=(15, 5))
    for i, image_path in enumerate(random_image_paths):
        img = mpimg.imread(image_path)
        ax[i].imshow(img)
        ax[i].axis("off")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, extract, and organize data.")
    parser.add_argument("zip_path", type=str, help="Path to the zip file.")
    
    args = parser.parse_args()

    base_dir = "data"
    set_names = ["train", "validation", "test"]
    
    download_extract_data(args.zip_path, base_dir)
    set_dirs = create_directories(base_dir, set_names)

    data_dir = Path(base_dir)
    images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
    labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]

    split_and_copy_data(images, labels, set_dirs)

    # Display random images for verification
    view_random_images(set_dirs["train"])
