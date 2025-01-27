import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score


class GunshotDetectionTrainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            optimizer,
            num_epochs,
            eval_metric,
            criterion,
            mean,
            std,
            device='cuda' if th.cuda.is_available() else 'cpu',
            patience=3
    ):
        """
        A helper class to train a gunshot detection model.

        Args:
            model (nn.Module): Your PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            num_epochs (int): Number of training epochs.
            eval_metric (str): Metric to track (currently supports only 'f1').
            criterion (nn.Module): Loss function (e.g., nn.BCELoss).
            mean (Tensor): Mean tensor for normalization.
            std (Tensor): Std tensor for normalization.
            device (str): 'cuda' or 'cpu'.
            patience (int): Early stopping patience for no improvement.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.eval_metric = eval_metric
        self.criterion = criterion
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.device = device
        self.patience = patience

        # For plotting/tracking
        self.train_losses = []
        self.valid_losses = []
        self.roc_aucs = []
        self.pr_aucs = []

        # Best metrics so far
        self.best_score = 0.0
        self.best_threshold = 0.5
        self.epochs_since_improvement = 0

        # Learning rate scheduler for plateau detection
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=2, verbose=True
        )

    def train_one_epoch(self, epoch):
        """Train the model for one epoch on the training dataset."""
        self.model.train()
        running_loss = 0.0

        # Create a progress bar
        train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}] Training")

        for features, labels, _ in train_loader_tqdm:
            # Move data to device
            features, labels = features.to(self.device), labels.to(self.device).float()

            # Normalize features
            features = (features - self.mean) / self.std

            # Forward and backward pass
            self.optimizer.zero_grad()
            outputs = self.model(features).view(-1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}], Training Loss: {epoch_loss:.4f}")
        return epoch_loss

    def validate_one_epoch(self, epoch):
        """Validate the model for one epoch on the validation dataset."""
        self.model.eval()
        valid_loss = 0.0

        with th.no_grad():
            for features, labels, _ in self.valid_loader:
                features, labels = features.to(self.device), labels.to(self.device).float()
                features = (features - self.mean) / self.std

                outputs = self.model(features).view(-1)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item() * features.size(0)

        valid_loss /= len(self.valid_loader.dataset)
        self.valid_losses.append(valid_loss)
        print(f"Epoch [{epoch + 1}], Validation Loss: {valid_loss:.4f}")
        return valid_loss

    def evaluate(self, threshold=0.5):
        """
        Evaluate the model on the validation dataset using self.eval_metric.
        Returns: eval_score, threshold, roc_auc, pr_auc, failed_samples
        """
        self.model.eval()
        y_true, y_pred, y_scores = [], [], []
        failed_samples = []

        with th.no_grad():
            for batch_idx, (features, labels, waveform) in enumerate(self.valid_loader):
                features, labels = features.to(self.device), labels.to(self.device).float()
                features = (features - self.mean) / self.std

                outputs = self.model(features).view(-1)
                predictions = (outputs >= threshold).float()

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
                y_scores.extend(outputs.cpu().tolist())

                # Track failed samples for analysis
                for i, (label, prediction, wav) in enumerate(zip(labels, predictions, waveform)):
                    if label != prediction:
                        failed_samples.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'label': label.item(),
                            'prediction': prediction.item(),
                            'spectrogram': features[i],  # normalized spectrogram
                            'waveform': wav
                        })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Evaluate metric
        if self.eval_metric == 'f1':
            eval_score = f1_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self.eval_metric}")

        # Calculate AUC-ROC and PR-AUC
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        return eval_score, threshold, roc_auc, pr_auc, failed_samples

    def train(self):
        """
        Main training loop. Tracks training and validation losses, and computes
        evaluation metrics (F1, AUC-ROC, PR-AUC). Implements early stopping using
        self.patience.
        Returns:
            (best_threshold, best_score, last_failed_samples)
        """
        last_failed_samples = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)

            eval_score, optimal_threshold, roc_auc, pr_auc, failed_samples = self.evaluate()
            self.roc_aucs.append(roc_auc)
            self.pr_aucs.append(pr_auc)
            last_failed_samples = failed_samples

            print(f"Epoch [{epoch + 1}], AUC-ROC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

            # Check improvement
            if eval_score > self.best_score:
                self.best_score = eval_score
                self.best_threshold = optimal_threshold
                self.epochs_since_improvement = 0
                print(f"New best {self.eval_metric}: {self.best_score:.4f}, model saved.")
                # You can save the model here if desired:
                # th.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.epochs_since_improvement += 1
                if self.epochs_since_improvement >= self.patience:
                    print(f"No improvement in {self.eval_metric} for {self.patience} epochs. Stopping training.")
                    break

            # Update LR if no improvement
            self.scheduler.step(eval_score)

        self._plot_metrics()
        return self.best_threshold, self.best_score, last_failed_samples

    def _plot_metrics(self):
        """Plot training and validation loss, plus AUC-ROC and PR-AUC."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, marker='o', label='Training Loss')
        plt.plot(range(1, len(self.valid_losses) + 1), self.valid_losses, marker='s', label='Validation Loss')
        plt.plot(range(1, len(self.roc_aucs) + 1), self.roc_aucs, marker='^', label='AUC-ROC')
        plt.plot(range(1, len(self.pr_aucs) + 1), self.pr_aucs, marker='v', label='PR-AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Training and Validation Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
