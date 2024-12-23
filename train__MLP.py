import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import pytorch_lightning as pl
from model_relu import Model  # Import your model here


def dataframe_to_numpy(df):
    # Create a numpy array with zeros, matching the shape of the dataframe
    np_data = np.zeros((df.shape[0], b), dtype='float32')

    # Iterate over each row in the dataframe
    for i, row in enumerate(df.iterrows()):
        # Extract the row index and data from the row object
        _, data = row

        # Assign the values of the row to the corresponding row in the numpy array
        np_data[i] = data.values
    return np_data

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

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

        #self.log('val_loss_step', loss)  # Log the loss for each validation step

        # Log the validation loss with the key 'val_loss'
        self.log('val_loss', loss)
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
        optimizer = optim.Adam(self.parameters())
        return optimizer

if __name__ == "__main__":
    # Load data and create datasets
    data = pd.read_csv(r'C:\Users\LENOVO\Downloads\RCdrone_features_15classes.csv')
    train, val, test = np.split(data.sample(frac=1), [int(0.8 * len(data)), int(0.9 * len(data))])

    # Convert DataFrame to NumPy array
    b = train.shape[1]
    train_np = dataframe_to_numpy(train)
    val_np = dataframe_to_numpy(val)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(train.iloc[:, 1:].values.astype(np.float32)),
        torch.from_numpy(train.iloc[:, 0].values).long()
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val.iloc[:, 1:].values.astype(np.float32)),
        torch.from_numpy(val.iloc[:, 0].values).long()
    )

    batch_size = 512
    train_loader =DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Select the GPU device index
    device_idx = 0
    # Check if the selected GPU device is available
    device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')

    model = Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    lightning_model = LightningModel()
    NUM = 10

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Directory where checkpoints will be saved
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint filename format with ".pth" extension
        monitor='val_loss',  # Metric to monitor for checkpointing
        save_top_k=1,  # Save only the best checkpoint (based on the monitored metric)
        mode='min',  # 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
    )

    # Create Trainer and start training, passing the checkpoint_callback
    trainer = pl.Trainer(
        max_epochs=NUM,
        callbacks=[checkpoint_callback]  # Pass the checkpoint_callback here
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    ##########################find and plot accuracy##################################
    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    common_epochs = min(len(train_accuracies), len(val_accuracies))

    train_accuracies = train_accuracies[:common_epochs]
    val_accuracies = val_accuracies[:common_epochs]


    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


    plt.figure()
    window_size = 15
    smoothed_train_accuracies = moving_average(train_accuracies, window_size)
    smoothed_val_accuracies = moving_average(val_accuracies, window_size)

    plt.plot(range(window_size, len(train_accuracies) + window_size), val_accuracies)
    plt.plot(range(window_size, len(val_accuracies) + window_size), train_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Accuracy plot')
    plt.savefig('Accuracy_plot.png')
    plt.show()

    #########################Compute and plot  loss#####################################"
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    train_losses = train_losses[:common_epochs]
    val_losses = val_losses[:common_epochs]


    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


    plt.figure()
    window_size = 15
    train_losses = moving_average(train_losses, window_size)
    val_losses = moving_average(val_losses, window_size)

    plt.plot(range(window_size, len(train_losses) + window_size), val_losses)
    plt.plot(range(window_size, len(val_losses) + window_size), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss plot')
    plt.savefig('loss_plot.png')
    plt.show()

