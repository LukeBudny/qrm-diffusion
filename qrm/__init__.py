# qrm/__init__.py

# Import core QRM components
from .qrm_trainer import QRMTrainer
from .qrm_dataloader import CachedCOCOIterable,COCOPromptDataset
from .qrm_models import QRMMLP, QRMTransformer

# Training-only utilities are imported lazily to avoid circular deps.
def _trainer():
    from .qrm_trainer_batches import QRMTrainer_batches
    return QRMTrainer_batches

# Define what is available when importing the QRM module
__all__ = ["QRMTrainer","CachedCOCOIterable","COCOPromptDataset","QRMMLP","QRMTransformer"]