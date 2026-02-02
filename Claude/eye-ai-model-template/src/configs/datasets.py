"""
Dataset configurations for RETFound model training.
This module defines the dataset collections used in different experiments.
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig

# Dataset from the notebook
# The notebook uses dataset RID: 2-A0JN with version 1.0.0
glaucoma_dataset = [DatasetSpecConfig(rid="2-A0JN", version="1.0.0")]

# Alternative dataset versions or combinations
# Add more datasets as needed for different experiments
alternative_dataset = [DatasetSpecConfig(rid="2-A0JN", version="2.0.0")]

# Multimodal datasets from the Jupyter notebook (Cell 5)
# These are used for the multimodal glaucoma classification workflow
multimodal_datasets = [
    DatasetSpecConfig(rid="4-4116", materialize=True),  # Selected images for training
    DatasetSpecConfig(rid="4-411G", materialize=True),  # Selected images for testing  
    DatasetSpecConfig(rid="2-7P5P", materialize=True),  # Full multimodal dataset
]

# Store the configurations
datasets_store = store(group="datasets")

# Default dataset
datasets_store(glaucoma_dataset, name="default_dataset")

# Alternative dataset version
datasets_store(alternative_dataset, name="version_2")

# Multimodal workflow datasets
datasets_store(multimodal_datasets, name="multimodal_dataset")

# Empty dataset (for testing without data)
datasets_store([], name="no_dataset")
