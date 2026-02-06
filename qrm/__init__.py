# qrm/__init__.py

# Import core QRM components
from .qrm_dataloader import CachedCOCOIterable
from .qrm_models import QRMModulatorLatent, QRMModulatorLatentV2,QRMModulatorLatentV3,QRMModulatorLatentV4

# Training-only utilities are imported lazily to avoid circular deps.
def _trainer():
    from .qrm_trainer_batches import QRMTrainer_batches
    return QRMTrainer_batches

# Define what is available when importing the QRM module
__all__ = ["CachedCOCOIterable","QRMModulatorLatent","QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4"]