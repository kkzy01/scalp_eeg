import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import random
from lightly.loss import SwaVLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes

# Using SSL to pre-train the ResNet model on the specific data, it helps to reduce the number of annotation needed
class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50()  # Load a ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head
        self.projection_head = SwaVProjectionHead(2048, 2048, 128)  # Define the projection head
        self.prototypes = SwaVPrototypes(128, 2048, 1)  # Define the prototypes module
        self.start_queue_at_epoch = 10  # Epoch to start using queue features
        self.queues = nn.ModuleList([MemoryBankModule(size=512) for _ in range(2)])  # Initialize memory bank modules
        self.criterion = SwaVLoss(sinkhorn_epsilon=0.05)  # Define the SwaV loss

    def training_step(self, batch, batch_idx):
        views = batch[0]  # Extract views from the batch
        high_resolution, low_resolution = views[:2], views[2:]  # Split into high and low-resolution views
        self.prototypes.normalize()  # Normalize prototypes

        # Extract features for high and low-resolution views
        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        # Compute prototypes for high and low-resolution features
        high_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features)  # Get queue prototypes if applicable
        loss = self.criterion(
            high_resolution_prototypes, low_resolution_prototypes, queue_prototypes
        )  # Compute SWAV loss
        self.log("swav_loss", loss)  # Log the loss
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)  # Define the optimizer
        return optim

    def _subforward(self, input):
        """
        Forward pass through the backbone and projection head.
        Args:
            input: Input tensor
        
        Returns:
            features: Normalized features
        """
        features = self.backbone(input).flatten(start_dim=1)  # Extract and flatten features from the backbone
        features = self.projection_head(features)  # Pass through the projection head
        features = nn.functional.normalize(features, dim=1, p=2)  # Normalize features
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):
        """
        Get queue prototypes for high-resolution features.
        Args:
            high_resolution_features: List of high-resolution feature tensors
        
        Returns:
            queue_prototypes: List of queue prototypes or None if before start_queue_at_epoch
        """
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Collect queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)  # Update queue with new features
            features = torch.permute(features, (1, 0))  # Permute for compatibility
            queue_features.append(features)

        # If still before the queue start epoch, just return None
        if (
            self.start_queue_at_epoch > 0
            and self.current_epoch < self.start_queue_at_epoch
        ):
            return None

        # Assign prototypes to queue features
        queue_prototypes = [
            self.prototypes(x, self.current_epoch) for x in queue_features
        ]
        return queue_prototypes