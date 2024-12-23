import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from model_relu import Model

train_losses = []
train_accuracies = []


class LightningModel(pl.LightningModule):
    def __init__(self):
        super(LightningModel, self).__init__()
        self.model = Model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        self.log('train_loss_step', loss)  # Log the loss for each training step
        self.log('train_accuracy_step', acc)
        # Ajouter les valeurs à la liste
        train_losses.append(loss.item())
        train_accuracies.append(acc)
        return {'loss': loss, 'accuracy': acc}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

        self.log('val_loss_step', loss)  # Log the loss for each validation step
        self.log('val_accuracy_step', acc)
        # Ajouter les valeurs à la liste
        val_losses.append(loss.item())
        val_accuracies.append(acc)
        return {'loss': loss, 'accuracy': acc}

    def validation_epoch_end(self, outputs):
        avg_train_loss = sum([x['loss'] for x in outputs]) / len(outputs)
        avg_train_acc = torch.tensor([x['accuracy'] for x in outputs]).mean()

        self.log('avg_val_loss', avg_train_loss)
        self.log('avg_val_accuracy', avg_train_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer




# Load data and create datasets
data = pd.read_csv(r'C:\Users\LENOVO\Downloads\RCdrone_features_15classes.csv')
train_size = int(0.75 * len(data))
val_size = int(0.15 * len(data))
train, val = np.split(data.sample(frac=1), [train_size])

train_dataset = TensorDataset(
    torch.from_numpy(train.iloc[:, 1:].values.astype(np.float32)),
    torch.from_numpy(train.iloc[:, 0].values).long()
)

val_dataset = TensorDataset(
    torch.from_numpy(val.iloc[:, 1:].values.astype(np.float32)),
    torch.from_numpy(val.iloc[:, 0].values).long()
)

batch_size = 128
num_epochs = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Select the GPU device index
device_idx = 0
# Check if the selected GPU device is available
device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')

# Define the optimizers
optimizers = [
    ("SGD", optim.SGD),
    ("Adagrad", optim.Adagrad),
    ("Adam", optim.Adam),
    ("Adamax", optim.Adamax),
    ("RMSprop", optim.RMSprop)
]

val_accuracies_per_optimizer = []
val_losses_per_optimizer = []

for optimizer_name, optimizer_class in optimizers:
    print(f"Training and evaluating using {optimizer_name} optimizer")

    # Create the model instance
    model = Model().to(device)

    if optimizer_name == "SGD":
        optimizer = optimizer_class(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adagrad":
        optimizer = optimizer_class(model.parameters(), lr=0.01, lr_decay=0.001)
    elif optimizer_name == "Adam":
        optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif optimizer_name == "Adamax":
        optimizer = optimizer_class(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)  # Paramètres spécifiques à Adamax
    elif optimizer_name == "RMSprop":
        optimizer = optimizer_class(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08)  # Paramètres spécifiques à RMSprop

    val_accuracies = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / val_total)

        print(f"{optimizer_name} - Epoch {epoch + 1} Validation Accuracy:", val_accuracy)

    val_accuracies_per_optimizer.append(val_accuracies)
    val_losses_per_optimizer.append(val_losses)

# ... (rest of your code)

# Plot the validation accuracies for all optimizers
plt.figure(figsize=(10, 8))
for optimizer_name, _ in optimizers:
    optimizer_index = [name for name, _ in optimizers].index(optimizer_name)
    val_accuracy = val_accuracies_per_optimizer[optimizer_index]
    plt.plot(range(1, num_epochs + 1), val_accuracy, marker='o', label=f"{optimizer_name} Validation Accuracy")

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison for Different Optimizers')
plt.legend()
plt.grid(True)
plt.savefig("val_accuracies.png")
plt.show()

# Plot the validation losses for all optimizers
plt.figure(figsize=(10, 8))
for optimizer_name, _ in optimizers:
    optimizer_index = [name for name, _ in optimizers].index(optimizer_name)
    val_loss = val_losses_per_optimizer[optimizer_index]
    plt.plot(range(1, num_epochs + 1), val_loss, marker='o', label=f"{optimizer_name} Validation Loss")

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison for Different Optimizers')
plt.legend()
plt.grid(True)
plt.savefig("val_losses.png")
plt.show()