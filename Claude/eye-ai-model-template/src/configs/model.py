"""
Model configuration registrations for RETFound training.
This module defines model configurations for the RETFound glaucoma classification model
and registers them into Hydra's store under the "model_config" group.
"""
from hydra_zen import builds, store
from models.retfound_model import retfound_model
from models.multimodal_glaucoma import multimodal_glaucoma

# Build the base configuration
RETFoundConfig = builds(
    retfound_model,
    # Model architecture
    model_type="retfound_finetune",
    num_classes=3,  # GS, POAG, PACG
    global_pool=True,
    
    # Training parameters
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-4,
    
    # Data splits
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_workers=4,
    
    # Sampler
    use_weighted_sampler=True,
    
    # Early stopping
    patience=10,
    min_delta=0.001,
    
    # Class weights (None for equal weighting)
    class_weights=None,
    
    # Output options
    save_model=True,
    generate_plots=True,
    
    populate_full_signature=True,
    zen_partial=True,  # We'll add the execution config later
)

# Register configurations
model_store = store(group="model_config")

# Default configuration
model_store(RETFoundConfig, name="default_model")

# Binary classification variant (GS vs POAG+PACG)
model_store(
    RETFoundConfig,
    name="binary_classification",
    num_classes=2,
    class_weights=[1.0, 1.5],  # Weight glaucoma class more
)

# High learning rate variant
model_store(
    RETFoundConfig,
    name="high_lr",
    learning_rate=5e-4,
    weight_decay=5e-4,
)

# Large batch variant
model_store(
    RETFoundConfig,
    name="large_batch",
    batch_size=64,
    learning_rate=2e-4,  # Scale LR with batch size
)

# Long training variant
model_store(
    RETFoundConfig,
    name="long_training",
    epochs=100,
    patience=20,
)

# Balanced sampling variant
model_store(
    RETFoundConfig,
    name="balanced_sampling",
    use_weighted_sampler=False,
    class_weights=[1.0, 2.0, 2.0],  # Compensate with loss weights
)

# Small dataset variant (80/10/10 split)
model_store(
    RETFoundConfig,
    name="small_data",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    batch_size=16,
    epochs=30,
)

# Fine-tuning variant (assumes pretrained model)
model_store(
    RETFoundConfig,
    name="fine_tune",
    learning_rate=1e-5,
    weight_decay=1e-5,
    epochs=20,
    batch_size=16,
)

# Quick test configuration
model_store(
    RETFoundConfig,
    name="quick_test",
    epochs=5,
    batch_size=8,
    patience=3,
    save_model=False,
    generate_plots=False,
)


# ============================================================================
# MULTIMODAL GLAUCOMA CONFIGURATIONS
# ============================================================================

# Build the multimodal glaucoma base configuration
MultimodalGlaucomaConfig = builds(
    multimodal_glaucoma,
    learning_rate=5e-3,
    epochs=100,
    batch_size=16,
    weight_decay=0.05,
    layer_decay=0.65,
    drop_path=0.2,
    validation_split=0.12,
    seed=12,
    populate_full_signature=True,
    zen_partial=True,
)

# Default multimodal configuration
model_store(MultimodalGlaucomaConfig, name="multimodal_default")

# Quick test configuration
model_store(
    MultimodalGlaucomaConfig,
    name="multimodal_test",
    epochs=5,
)

# Configuration with higher learning rate for faster convergence
model_store(
    MultimodalGlaucomaConfig,
    name="multimodal_fast",
    learning_rate=1e-2,
    epochs=50,
)

# Configuration with stronger regularization
model_store(
    MultimodalGlaucomaConfig,
    name="multimodal_regularized",
    weight_decay=0.1,
    drop_path=0.3,
    validation_split=0.15,
)

# Configuration for larger batch size
model_store(
    MultimodalGlaucomaConfig,
    name="multimodal_large_batch",
    learning_rate=1e-2,
    batch_size=32,
)

# Configuration for extended training
model_store(
    MultimodalGlaucomaConfig,
    name="multimodal_extended",
    epochs=150,
    learning_rate=3e-3,
)
