import os
import torch

class TrainingLogger:
    def __init__(self, file_name="training.log"):
        """
        Initialize the TrainingLogger.

        Args:
            file_name (str): Path to the log file where training information will be saved.
        """
        self.file_name = file_name
        self.best_epoch = None
        self.best_val_loss = float('inf')
        self.train_loss = None
        # Create a new log file or clear the existing one
        with open(self.file_name, "w") as file:
            file.write("Epoch,Training Loss,Validation Loss,Best Epoch\n")

    def log(self, epoch, train_loss, val_loss):
        """
        Log training and validation metrics for a specific epoch.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Training loss for the epoch.
            val_loss (float): Validation loss for the epoch.
        """
        self.train_loss = train_loss
        # Update the best epoch if the current validation loss is lower
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        # Write log entry to the file
        with open(self.file_name, "a") as file:
            file.write(f"{epoch},{self.train_loss},{val_loss:.4f},{self.best_epoch}\n")

    def print_summary(self):
        """
        Print a summary of the best epoch and validation loss.
        """
        print("\nTraining Summary:")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Epoch Training Loss:{self.train_loss}")
        print(f"Best Validation Loss: {self.best_val_loss}")

class ModelSaver:
    def __init__(self, save_dir, best_model_name="best_model.pth"):
        self.save_dir = save_dir
        self.best_model_name = best_model_name
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, model, epoch, is_best=False):
        """
        Save the model checkpoint.
        :param model: PyTorch model to save
        :param epoch: Current epoch
        :param validation_loss: Validation loss at the current epoch
        :param is_best: Flag to indicate if the current model is the best
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        file_path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        torch.save(checkpoint, file_path)
        print(f"Model checkpoint saved at {file_path}")

        if is_best:
            best_path = os.path.join(self.save_dir, self.best_model_name)
            torch.save(checkpoint, best_path)
            print(f"Best model updated at {best_path} at epoch {epoch}")