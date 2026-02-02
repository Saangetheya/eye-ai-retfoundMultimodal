"""
Asset configurations for RETFound model.
This module defines the pretrained model weights and other assets needed for execution.
"""
from hydra_zen import store
from deriva_ml.execution import AssetRIDConfig

# RETFound pretrained weights from the notebook
# The notebook uses RETFound_cfp_weights.pth (RID: 2-A0MR)
retfound_weights = [AssetRIDConfig("2-A0MR")]

# Alternative asset configuration if multiple weights are needed
# You can add more RIDs here as needed
alternative_weights = [
    AssetRIDConfig("2-A0MR"),  # RETFound CFP weights
    # Add other asset RIDs here if needed
]

# Multimodal workflow asset (from Jupyter notebook Cell 5)
# RETFound pretrained weights used for multimodal classification
multimodal_asset = [AssetRIDConfig("4-S4TJ")]

# Store the configurations
asset_store = store(group="assets")

# Default asset configuration
asset_store(retfound_weights, name="default_asset")

# Alternative weights
asset_store(alternative_weights, name="retfound_cfp")

# Multimodal workflow asset
asset_store(multimodal_asset, name="multimodal_asset")

# Empty asset configuration (for training from scratch)
asset_store([], name="no_pretrained")
