import os
import shutil

# Define the paths for your original dataset and the split datasets
original_dataset_dir = "dataset"
base_dir = "datasetsplit"

# Create directories for training, validation, and test sets
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, "train")
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, "validation")
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, "test")
os.makedirs(test_dir, exist_ok=True)

from random import shuffle

# Define the subdirectories for each class
class_names = os.listdir(original_dataset_dir)

# Set the split ratios (80% train, 10% validation, 10% test)
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

for class_name in class_names:
    class_dir = os.path.join(original_dataset_dir, class_name)
    image_list = os.listdir(class_dir)
    shuffle(image_list)  # Shuffle the list of images

    # Calculate split indices
    num_images = len(image_list)
    num_train = int(num_images * train_ratio)
    num_validation = int(num_images * validation_ratio)

    # Split the images into train, validation, and test sets
    train_images = image_list[:num_train]
    validation_images = image_list[num_train : num_train + num_validation]
    test_images = image_list[num_train + num_validation :]

    # Create subdirectories in the split sets
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)

    validation_class_dir = os.path.join(validation_dir, class_name)
    os.makedirs(validation_class_dir, exist_ok=True)

    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy or move images to their respective directories
    for image_name in train_images:
        src = os.path.join(class_dir, image_name)
        dst = os.path.join(train_class_dir, image_name)
        shutil.copy(src, dst)

    for image_name in validation_images:
        src = os.path.join(class_dir, image_name)
        dst = os.path.join(validation_class_dir, image_name)
        shutil.copy(src, dst)

    for image_name in test_images:
        src = os.path.join(class_dir, image_name)
        dst = os.path.join(test_class_dir, image_name)
        shutil.copy(src, dst)
