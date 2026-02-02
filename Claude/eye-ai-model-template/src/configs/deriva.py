"""
DerivaML configurations.
This module defines the Deriva catalog connection configurations.
"""
from hydra_zen import store
from deriva_ml import DerivaMLConfig

# Store the configurations
deriva_store = store(group="deriva_ml")

# Production configuration (from the notebook)
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="www.eye-ai.org",
    catalog_id="eye-ai",
)

# Localhost configuration for development
deriva_store(
    DerivaMLConfig,
    name="localhost",
    hostname="localhost",
    catalog_id=2,
    use_minid=False,
)
