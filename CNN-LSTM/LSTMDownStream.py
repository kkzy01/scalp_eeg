import torch.nn.functional as F
from torch import nn
import torch
import sklearn
import pytorch_lightning as pl
import sys
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, unpad_sequence

sys.path.append('../tools')
from sigmoid_loss import sigmoid_focal_loss


# load the ResNet trained with SSL to use with LSTM for final prediction in this way
# swav = SwaV().load_from_checkpoint('...')
# model = SupervisedDownstream(swav.backbone)
# sd = SupervisedDownstream(model)
# ...


# Define the supervised downstream model for downstream classification
class SupervisedDownstream(pl.LightningModule):
    def __init__(self, backbone, emb_len=2048, unfreeze_backbone_at_epoch=100):
        super().__init__()
        self.backbone = backbone  # Pre-trained backbone model
        self.emb_len = emb_len  # Embedding length from the backbone model
        self.fc1 = nn.Linear(self.emb_len, 256)  
        self.fc2 = nn.Linear(256, 64)  
        self.dp = nn.Dropout1d(p=0.2)  
        self.fc3 = nn.Linear(64, 2)  
        self.softmax = nn.Softmax(dim=1)  
        self.alpha = 0  # Alpha parameter for focal loss
        self.gamma = 5  # Gamma parameter for focal loss
        self.lstm = nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)  # Bi-directional LSTM
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch  # Epoch to unfreeze backbone for fine-tuning
        self.enable_mc_dropout = False  # Flag for enabling MC Dropout for uncertainty estimation

    def forward(self, x, seq_len):
        """
        Forward pass through the model.
        Args:
            x: Input tensor of shape (Batch, 1, emb_len)
            seq_len: List of sequence lengths in the batch
        
        For example, input x has a batch size of 55, that may contain 4 different EEG clips with varying length of size (10, 20, 15, 10), then seqlen =  [10, 20, 15, 10]
        self.set_requires_grad(self.backbone, False)
        
        Returns:
            pred: Predictions
            emb: Embeddings from the backbone
            emb_t: LSTM-transformed embeddings
        """
        # self.set_requires_grad(self.backbone, False)  # Freeze backbone weights

        # Use backbone model in eval mode for frozen layers
        if self.current_epoch < self.unfreeze_backbone_at_epoch:
            self.backbone.eval()
            x = self.backbone(x)
            with torch.no_grad():
                emb = x.view(-1, self.emb_len)
        else:
            x = self.backbone(x)
            emb = x.view(-1, self.emb_len)

        # Apply dropout in training mode
        if self.enable_mc_dropout:
            self.dp.train()
        else:
            self.dp.eval()

        x = F.relu(self.fc1(emb))
        x = self.dp(x)

        # convert large batch (2D array) of single second window in to small batch (3D array) of sequences for LSTM
        x = torch.split(x, seq_len, dim=0)  # Split tensor into sequences
        x = pack_sequence(x, enforce_sorted=False)  # Pack sequences for LSTM
        x, (_, _) = self.lstm(x)  # Pass through LSTM

        # convert back to 2D array for fc layers
        x, out_len = pad_packed_sequence(x, batch_first=True)  # Pad sequences back
        emb_t = torch.concat(unpad_sequence(x, out_len, batch_first=True))  # Unpad sequences

        x = self.dp(emb_t)

        x = F.relu(self.fc2(x))
        pred = self.fc3(x)  # Final predictions before softmax

        return pred, emb, emb_t

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.
        Args:
            batch: Tuple of input tensor, labels, and sequence lengths
            batch_idx: Batch index
        
        Returns:
            loss: Calculated loss for the batch
        """
        x, y, seq_len = batch # you OWN Dataloader and collate_fn NEEDED
        pred, _, _ = self.forward(x, seq_len)
        pred = self.softmax(pred)  # Apply softmax to predictions
        label = F.one_hot(y, num_classes=2).squeeze()  # One-hot encode labels
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean') # focal loss here for unbalanced label, use Crossentropy also works
        self.log("train_loss", loss)  # Log training loss with lightning
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch.
        Args:
            batch: Tuple of input tensor, labels, and sequence lengths
            batch_idx: Batch index
        
        Returns:
            Tuple of predictions and labels
        """
        x, y, seq_len = batch
        pred, _, _ = self.forward(x, seq_len)
        pred = self.softmax(pred)  # Apply softmax to predictions
        label = F.one_hot(y, num_classes=2).squeeze()  # One-hot encode labels
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        out = torch.argmax(pred, dim=1)  # Get predicted class

        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
            out, target, labels=[0, 1], zero_division=0
        )
        acc = sklearn.metrics.accuracy_score(out, target)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        self.log("val_precision", precision[1], prog_bar=False)
        self.log("val_recall", recall[1], prog_bar=False)
        self.log("val_f1", fscore[1], prog_bar=False)
        return pred, label

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for each batch.
        Args:
            batch: Tuple of input tensor, labels, and sequence lengths
            batch_idx: Batch index
        
        Returns:
            Tuple of predictions, labels, embeddings, LSTM embeddings, and sequence lengths
        """
        x, y, seq_len = batch
        pred, emb, emb_t = self(x, seq_len)
        return pred, y, emb, emb_t, seq_len

    def configure_optimizers(self):
        """
        Configure optimizers for training.
        
        Returns:
            optimizer: Configured optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_requires_grad(self, model, requires_grad=True, exclude=None):
        """
        Set requires_grad attribute of model parameters.
        Args:
            model: Model whose parameters need to be updated
            requires_grad: Flag to set requires_grad
            exclude: Layers to exclude from updating requires_grad, allow partial frozen of ResNet
        
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

        if exclude is not None:
            for name, child in model.named_children():
                if name in exclude:
                    for param in child.parameters():
                        param.requires_grad = not requires_grad
