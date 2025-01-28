from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import DirectoryIterator
import numpy as np
import os

# Define augmentation for minority classes
minority_augmentation = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,             # Rotate images by up to 40 degrees
    width_shift_range=0.2,         # Shift images horizontally by up to 20%
    height_shift_range=0.2,        # Shift images vertically by up to 20%
    shear_range=0.2,               # Apply shear transformations
    zoom_range=0.3,                # Zoom in/out by up to 30%
    horizontal_flip=True,          # Flip images horizontally
    fill_mode='nearest'            # Fill missing pixels with the nearest value
)

# Define augmentation for majority classes (less aggressive)
majority_augmentation = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.2,                # Apply minimal zoom
    horizontal_flip=True           # Flip images horizontally
)


def balanced_flow_from_directory(directory, target_size, batch_size, class_mode, shuffle=True):
    # Create a standard generator to get class distributions
    base_generator = ImageDataGenerator(rescale=1.0/255)
    base_dataset = base_generator.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )
    
    # Get class counts
    class_counts = np.bincount(base_dataset.classes)
    max_class_count = np.max(class_counts)
    
    # Define augmentation for each class
    augmented_datasets = []
    for class_idx, class_name in enumerate(base_dataset.class_indices):
        class_dir = os.path.join(directory, class_name)  # Ensure this is a string path
        if class_counts[class_idx] < max_class_count:
            # Minority class: apply aggressive augmentation
            minority_generator = minority_augmentation.flow_from_directory(
                directory,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                classes=[class_name],  # Pass the class name as a string
                shuffle=False
            )
            augmented_datasets.append(minority_generator)
        else:
            # Majority class: apply minimal augmentation
            majority_generator = majority_augmentation.flow_from_directory(
                directory,
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                classes=[class_name],  # Pass the class name as a string
                shuffle=False
            )
            augmented_datasets.append(majority_generator)
    
    # Combine datasets
    balanced_dataset = DirectoryIterator(
        directory,
        augmented_datasets,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )
    return balanced_dataset