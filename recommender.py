import tensorflow as tf
import numpy as np
import keras
import os
import pickle  # Added import for pickle module

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


def load_model():
    # Load MobileNetV2 as the feature extraction model
    base_model = MobileNetV2(weights="imagenet", include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded_dims)
    features = model.predict(img_preprocessed)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features


# Directory containing images to analyze
image_dir = "D:/fashion_styles/dataset"

model = load_model()

# Check if the serialized features file exists
features_file_path = "image_features.pkl"

if os.path.exists(features_file_path):
    # Load image features from the file
    with open(features_file_path, "rb") as file:
        image_features = pickle.load(file)
else:
    # Extract features from all images and map them with the corresponding file paths
    image_features = {}
    for genre_folder in os.listdir(image_dir):
        genre_folder_path = os.path.join(image_dir, genre_folder)
        if os.path.isdir(genre_folder_path):
            for img_file in os.listdir(genre_folder_path):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    img_path = os.path.join(genre_folder_path, img_file)
                    features = extract_features(img_path, model)
                    image_features[img_path] = features

    # Save image features to a file
    with open(features_file_path, "wb") as file:
        pickle.dump(image_features, file)


def recommend(file_path):
    # Assuming we pick a sample image as the "liked" image for our sample user
    sample_user_liked_img_path = file_path
    model = load_model()
    sample_user_liked_features = extract_features(sample_user_liked_img_path, model)

    # Compute similarities between the "liked" image and all other images
    similarities = {}
    for img_path, features in image_features.items():
        similarity = cosine_similarity([sample_user_liked_features], [features])[0][0]
        similarities[img_path] = similarity

    # Sort images based on similarity score in descending order
    sorted_similar_images = sorted(
        similarities.items(), key=lambda item: item[1], reverse=True
    )

    # Get the top 5 most similar images (excluding the first one, which is the same image)
    top_similar_images = sorted_similar_images[1:6]
    # Extract filenames and directories
    file_info = [
        "/static/"
        + os.path.basename(os.path.dirname(item[0]))
        + "/"
        + os.path.basename(item[0])
        for item in top_similar_images
    ]
    return file_info


print(recommend("C:/Users/swaat/Downloads/3d8ed7e2ef6c10391dc59afcb4b8e238.jpg"))
