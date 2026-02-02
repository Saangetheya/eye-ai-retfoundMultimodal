"""
Multimodal Glaucoma Classification Model

This model applies computer vision to multimodal data for glaucoma prediction,
combining fundus images with clinical tabular data (MD, CDR, IOP, etc.).
"""

import torch
import pandas as pd
import numpy as np
import random
import shutil
import os
from pathlib import Path
import logging
from typing import Dict, Any

from eye_ai.eye_ai import EyeAI
from deriva_ml import DatasetBag

# Import the RETFound training functions
from main_finetune import main, get_args_parser


def multimodal_glaucoma(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    layer_decay: float,
    drop_path: float,
    validation_split: float,
    seed: int,
    ml_instance: EyeAI,
    execution: Any,
):
    """
    Train a multimodal glaucoma classification model using RETFound with fine-tuning.
    
    Args:
        learning_rate: Base learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        weight_decay: Weight decay for regularization
        layer_decay: Layer-wise learning rate decay
        drop_path: Drop path rate for regularization
        validation_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        ml_instance: EyeAI instance for catalog operations
        execution: Execution context containing datasets and assets
    """
    
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting multimodal glaucoma classification training")
    
    # Extract datasets and assets from execution context
    training_ds_bag = execution.datasets[0]
    testing_ds_bag = execution.datasets[1]
    multimodal_full_ds_bag = execution.datasets[2]
    retfound_pretrained_weight = execution.asset_paths['Execution_Asset'][0]
    
    # Create working directory
    working_dir = execution._working_dir / execution.execution_rid
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataframes from dataset bags
    logger.info("Preparing training and testing dataframes")
    train_df = get_dataframe_from_bag(training_ds_bag, multimodal_full_ds_bag, ml_instance)
    test_df = get_dataframe_from_bag(testing_ds_bag, multimodal_full_ds_bag, ml_instance)
    
    # Filter and prepare data
    req_cond = ['POAG', 'GS', 'PACG']
    
    # Prepare training data
    filtered_train_df = train_df[['RID_Image', 'Filename', 'Updated_Condition_Label', 
                                   'Condition_Display', 'MD']]
    train = filtered_train_df[filtered_train_df['Updated_Condition_Label'].isin(req_cond)]
    train['MD'] = pd.to_numeric(train['MD'], errors='coerce')
    train = train.dropna(subset=['MD'])
    
    # Prepare testing data
    filtered_test_df = test_df[['RID_Image', 'Filename', 'Updated_Condition_Label',
                                 'Condition_Display', 'MD', 'CDR', 'Subject_Gender',
                                 'Subject_Ethnicity', 'IOP', 'Disc_Area(mm^2)']]
    test = filtered_test_df[filtered_test_df['Updated_Condition_Label'].isin(req_cond)]
    test['MD'] = pd.to_numeric(test['MD'], errors='coerce')
    test = test.dropna(subset=['MD'])
    
    logger.info(f"Training samples: {len(train)}, Testing samples: {len(test)}")
    
    # Create dataset folders with stratification
    train_dir = create_dataset_folder(train, working_dir, "train")
    test_dir = create_dataset_folder(test, working_dir, "test")
    
    # Create validation split
    val_dir = working_dir / "val"
    create_validation_set(train_dir, val_dir, split_ratio=validation_split, seed=seed)
    
    # Create metadata for test set (for evaluation)
    create_meta_folder(test, working_dir)
    
    # Log dataset statistics
    train_counts = count_images_per_class(train_dir)
    val_counts = count_images_per_class(val_dir)
    test_counts = count_images_per_class(test_dir)
    
    logger.info("Training Set:")
    for class_name, count in train_counts.items():
        logger.info(f"  {class_name}: {count} images")
    logger.info("Validation Set:")
    for class_name, count in val_counts.items():
        logger.info(f"  {class_name}: {count} images")
    logger.info("Test Set:")
    for class_name, count in test_counts.items():
        logger.info(f"  {class_name}: {count} images")
    
    # Setup model output directory
    asset_path_output = execution.asset_file_path(
        filename="model",
        description="Multimodal glaucoma classification model outputs"
    )
    asset_path_output.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    logger.info("Starting model training with RETFound")
    with execution.execute() as exec:
        args_list = [
            "--model", "RETFound_mae",
            "--savemodel",
            "--global_pool",
            "--batch_size", str(batch_size),
            "--world_size", "1",
            "--epochs", str(epochs),
            "--blr", str(learning_rate),
            "--layer_decay", str(layer_decay),
            "--weight_decay", str(weight_decay),
            "--drop_path", str(drop_path),
            "--nb_classes", "3",
            "--data_path", str(working_dir),
            "--input_size", "224",
            "--task", str(asset_path_output),
            "--output_dir", str(asset_path_output),
            "--finetune", str(retfound_pretrained_weight),
        ]
        
        args = get_args_parser().parse_args(args_list)
        criterion = torch.nn.CrossEntropyLoss()
        
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        main(args, criterion)
    
    logger.info(f"Training completed. Model outputs saved to {asset_path_output}")
    
    return {
        "model_path": str(asset_path_output),
        "train_samples": len(train),
        "val_samples": sum(val_counts.values()),
        "test_samples": len(test),
        "working_dir": str(working_dir)
    }


def get_dataframe_from_bag(ds_bag: DatasetBag, multimodal_full_ds_bag: DatasetBag, 
                           ml_instance: EyeAI) -> pd.DataFrame:
    """
    Extract and merge all necessary tables from a dataset bag.
    
    Args:
        ds_bag: Dataset bag containing image selection
        multimodal_full_ds_bag: Full multimodal dataset bag
        ml_instance: EyeAI instance for multimodal operations
        
    Returns:
        Merged dataframe with all features
    """
    observation_table = ds_bag.get_table_as_dataframe('Observation')
    image_table = ds_bag.get_table_as_dataframe('Image')
    laterality_table = ds_bag.get_table_as_dataframe('Execution_Image_Fundus_Laterality')
    condition_table = ds_bag.get_table_as_dataframe('Execution_Subject_Chart_Label')
    
    image_table_filtered = image_table[['RID', 'Filename', 'Observation']].rename(
        columns={'RID': 'RID_Image'})
    laterality_table_filtered = laterality_table[['Image', 'Image_Side']].rename(
        columns={'Image': 'RID_Image'})
    condition_table_filtered = condition_table[['Subject', 'Condition_Label', 'Image_Side']].rename(
        columns={'Condition_Label': 'Updated_Condition_Label'})
    
    image_laterality = pd.merge(image_table_filtered, laterality_table_filtered, 
                                left_on='RID_Image', right_on='RID_Image', how='inner')
    observation_table_filtered = observation_table[['RID', 'Subject', 'Age']].rename(
        columns={'RID': 'RID_Observation'})
    image_laterality_observation = pd.merge(image_laterality, observation_table_filtered, 
                                           left_on='Observation', right_on='RID_Observation', 
                                           how='inner')
    image_laterality_observation_condition = pd.merge(condition_table_filtered, 
                                                     image_laterality_observation, 
                                                     left_on=['Subject', 'Image_Side'], 
                                                     right_on=['Subject', 'Image_Side'], 
                                                     how='inner')
    
    wide = ml_instance.multimodal_wide(multimodal_full_ds_bag)
    
    image_observation_laterality_subject_wide = pd.merge(
        wide,
        image_laterality_observation_condition,
        left_on=['RID_Subject', 'Image_Side'],
        right_on=['Subject', 'Image_Side'],
        how='inner'
    )
    
    return image_observation_laterality_subject_wide


def create_dataset_folder(df: pd.DataFrame, output_path: Path, output_name: str) -> Path:
    """
    Create a folder structure for training with images organized by class.
    
    Classes:
    - 0_GS: Glaucoma Suspect
    - 1_Mild: POAG/PACG with MD > -6
    - 2_Moderate_Severe: POAG/PACG with MD <= -6
    
    Args:
        df: Dataframe with image information
        output_path: Base output path
        output_name: Name of the output folder (train/test/val)
        
    Returns:
        Path to created dataset folder
    """
    output_path = output_path / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_path_0 = output_path / "0_GS"
    output_path_1 = output_path / "1_Mild"
    output_path_2 = output_path / "2_Moderate_Severe"
    
    output_path_0.mkdir(parents=True, exist_ok=True)
    output_path_1.mkdir(parents=True, exist_ok=True)
    output_path_2.mkdir(parents=True, exist_ok=True)
    
    for index, row in df.iterrows():
        src_path = row["Filename"]
        dest_name = row["RID_Image"] + ".jpg"
        dx = row['Updated_Condition_Label']
        md = row['MD']
        
        if dx == "GS":
            dest_path = output_path_0 / dest_name
        elif dx in ["POAG", "PACG"] and md > -6:
            dest_path = output_path_1 / dest_name
        elif dx in ["POAG", "PACG"] and md <= -6:
            dest_path = output_path_2 / dest_name
        else:
            continue
        
        shutil.copy2(src_path, dest_path)
    
    return output_path


def create_validation_set(train_dir: Path, val_dir: Path, split_ratio: float = 0.15,
                          seed: int = 12):
    """
    Create validation set by moving images from training set.
    
    Args:
        train_dir: Training directory with class folders
        val_dir: Validation directory to create
        split_ratio: Fraction of training data to use for validation
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    os.makedirs(val_dir, exist_ok=True)
    
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        
        if os.path.isdir(class_train_path):
            os.makedirs(class_val_path, exist_ok=True)
            
            images = [f for f in os.listdir(class_train_path) 
                     if os.path.isfile(os.path.join(class_train_path, f))]
            num_val = int(len(images) * split_ratio)
            
            val_images = random.sample(images, num_val)
            for img in val_images:
                shutil.move(os.path.join(class_train_path, img), 
                           os.path.join(class_val_path, img))


def create_meta_folder(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Create metadata CSV file for test set evaluation.
    
    Args:
        df: Dataframe with test set information
        output_path: Output directory path
        
    Returns:
        Path to output directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for index, row in df.iterrows():
        filename = row["RID_Image"] + ".jpg"
        metadata.append({
            "Filename": filename,
            "MD": row["MD"],
            "CDR": row["CDR"],
            "SEX": row['Subject_Gender'],
            "ETHNICITY": row['Subject_Ethnicity'],
            "IOP": row['IOP'],
            "AREA": row['Disc_Area(mm^2)'],
            "CONDITION": row['Updated_Condition_Label']
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path / "metadata_final.csv", index=False)
    
    return output_path


def count_images_per_class(directory: Path) -> Dict[str, int]:
    """
    Count number of images in each class folder.
    
    Args:
        directory: Directory containing class folders
        
    Returns:
        Dictionary mapping class names to image counts
    """
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) 
                            if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = num_images
    return class_counts
