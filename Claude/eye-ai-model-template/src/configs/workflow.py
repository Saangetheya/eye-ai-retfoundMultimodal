"""
Workflow configurations for RETFound model training.
"""
from hydra_zen import store, builds
from deriva_ml.execution import Workflow

# Create workflow configuration
WorkflowConf = builds(
    Workflow,
    name="RETFound Glaucoma Classification",
    workflow_type="Deep Learning - Image Classification",
    description="RETFound model for glaucoma classification (GS, POAG, PACG)",
    populate_full_signature=True,
)

# Store the configuration
workflow_store = store(group="workflow")
workflow_store(WorkflowConf, name="default_workflow")

# Alternative workflow for binary classification
workflow_store(
    WorkflowConf,
    name="binary_workflow",
    name="RETFound Binary Classification",
    description="RETFound model for binary glaucoma classification (GS vs POAG/PACG)",
)

# Multimodal workflow configuration
MultimodalWorkflowConf = builds(
    Workflow,
    name="Multimodal Glaucoma Classification",
    workflow_type="Deep Learning - Multimodal Classification",
    description="Multimodal glaucoma classification combining fundus images with clinical data (MD, CDR, IOP, etc.) using RETFound",
    populate_full_signature=True,
)

workflow_store(MultimodalWorkflowConf, name="multimodal_workflow")
