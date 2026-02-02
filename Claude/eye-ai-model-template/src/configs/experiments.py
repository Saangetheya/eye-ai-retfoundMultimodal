"""
Experiment definitions for RETFound model training.
These can be run with: python deriva_run.py --multirun +experiment=experiment_name
"""
from hydra_zen import make_config, store

# Get the base configuration
app_config = store[None]
app_name = next(iter(app_config))
deriva_model_config = store[None][app_name]

# Create experiment store
experiment_store = store(group="experiments")

# Experiment 1: Baseline 3-class classification
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "default_model"},
        ],
        bases=(deriva_model_config,)
    ),
    name="baseline_3class",
)

# Experiment 2: Binary classification
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "binary_classification"},
        ],
        bases=(deriva_model_config,)
    ),
    name="binary_classification",
)

# Experiment 3: Long training with high learning rate
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "high_lr"},
        ],
        bases=(deriva_model_config,)
    ),
    name="high_lr_experiment",
)

# Experiment 4: Large batch training
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "large_batch"},
        ],
        bases=(deriva_model_config,)
    ),
    name="large_batch_experiment",
)

# Experiment 5: Extended training
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "long_training"},
        ],
        bases=(deriva_model_config,)
    ),
    name="extended_training",
)

# Experiment 6: Fine-tuning experiment
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "fine_tune"},
        ],
        bases=(deriva_model_config,)
    ),
    name="fine_tuning",
)

# Experiment 7: Quick test run
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "retfound_cfp"},
            {"override /model_config": "quick_test"},
        ],
        bases=(deriva_model_config,)
    ),
    name="quick_test",
)

# Experiment 8: Training without pretrained weights
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "default_dataset"},
            {"override /assets": "no_pretrained"},
            {"override /model_config": "default_model"},
        ],
        bases=(deriva_model_config,)
    ),
    name="from_scratch",
)


# ============================================================================
# MULTIMODAL GLAUCOMA EXPERIMENTS
# ============================================================================

# Experiment 9: Multimodal glaucoma classification (default)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_default"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_default",
)

# Experiment 10: Quick multimodal test
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_test"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_quick_test",
)

# Experiment 11: Fast convergence multimodal
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_fast"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_fast_convergence",
)

# Experiment 12: Heavily regularized multimodal
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_regularized"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_regularized",
)

# Experiment 13: Large batch multimodal
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_large_batch"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_large_batch",
)

# Experiment 14: Extended training multimodal
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /datasets": "multimodal_dataset"},
            {"override /assets": "multimodal_asset"},
            {"override /model_config": "multimodal_extended"},
        ],
        bases=(deriva_model_config,)
    ),
    name="multimodal_extended_training",
)
