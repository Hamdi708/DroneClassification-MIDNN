import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from multiprocessing import freeze_support
from tqdm import tqdm
import numpy as np
import seaborn as sns
data_dir = r"C:\Users\LENOVO\Desktop\Data1"

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        residual = self.conv_residual(residual)
        out += residual
        out = F.relu(out)
        return out

class Classifier(pl.LightningModule):
    def __init__(self, class_number):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block = ResidualBlock(16, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm_input_size = 128
        self.lstm_hidden_size = 64
        self.lstm_layers = 1
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_layers, batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, class_number)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.residual_block(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        lstm_output, _ = self.lstm(x)
        x = lstm_output[:, -1, :]
        x = self.fc(x)
        return x

class CustomLightningModule(pl.LightningModule):
    def __init__(self, num_epochs, train_loader, val_loader, model, criterion, optimizer):
        super(CustomLightningModule, self).__init__()
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)  # Utilisation de train_accuracy_values ici
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)  # Utilisation de train_accuracy_values ici

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

        self.log('avg_val_loss', avg_val_loss)
        self.log('avg_val_accuracy', avg_val_acc)

        self.val_accuracies.append(avg_val_acc)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        return self.optimizer

if __name__ == '__main__':
    freeze_support()

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    number_classes = len(dataset.classes)

    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.1

    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Classifier(class_number=number_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    custom_lightning_module = CustomLightningModule(num_epochs=40, train_loader=train_loader, val_loader=val_loader,
                                                    model=model, criterion=criterion, optimizer=optimizer)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    trainer = pl.Trainer()

    for epoch in range(custom_lightning_module.num_epochs):
        print(f'Epoch [{epoch + 1}/{custom_lightning_module.num_epochs}]...')

        # Training
        custom_lightning_module.model.train()
        train_loss = 0.0
        train_accuracy_values = []

        train_loader_iter = tqdm(custom_lightning_module.train_loader, total=len(custom_lightning_module.train_loader),
                                 desc=f"Epoch {epoch + 1}/{custom_lightning_module.num_epochs} (Training)")

        for i, (images, labels) in enumerate(train_loader_iter):
            images = images.to(custom_lightning_module.device)
            labels = labels.to(custom_lightning_module.device)

            custom_lightning_module.optimizer.zero_grad()
            outputs = custom_lightning_module.model(images)
            loss = custom_lightning_module.criterion(outputs, labels)
            loss.backward()
            custom_lightning_module.optimizer.step()

            train_loss += loss.item()

            # Calcul de l'exactitude et ajout à la liste
            accuracy = accuracy_score(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
            train_accuracy_values.append(accuracy)

            train_loader_iter.set_postfix(
                {"Train Loss": loss.item()})  # Afficher la dernière exactitude

        average_train_loss = train_loss / len(custom_lightning_module.train_loader)
        average_train_accuracy = np.mean(train_accuracy_values)  # Exactitude moyenne pour cette époque

        # Validation
        custom_lightning_module.model.eval()
        val_loss = 0.0
        val_accuracy_values = []  # Liste pour stocker les exactitudes de validation de cette époque

        val_loader_iter = tqdm(custom_lightning_module.val_loader, total=len(custom_lightning_module.val_loader),
                               desc=f"Epoch {epoch + 1}/{custom_lightning_module.num_epochs} (Validation)")

        with torch.no_grad():
            for images, labels in val_loader_iter:
                images = images.to(custom_lightning_module.device)
                labels = labels.to(custom_lightning_module.device)

                outputs = custom_lightning_module.model(images)
                loss = custom_lightning_module.criterion(outputs, labels)
                val_loss += loss.item()

                # Calcul de l'exactitude et ajout à la liste
                accuracy = accuracy_score(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
                val_accuracy_values.append(accuracy)

                val_loader_iter.set_postfix({"Validation Loss": loss.item()})  # Afficher la dernière exactitude

        average_val_loss = val_loss / len(custom_lightning_module.val_loader)
        average_val_accuracy = np.mean(val_accuracy_values)  # Exactitude moyenne pour cette époque

        # Ajout des pertes et des exactitudes moyennes aux listes
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_accuracies.append(average_train_accuracy)
        val_accuracies.append(average_val_accuracy)

    print(train_accuracies)
    print(train_losses)
    print(val_accuracies)
    print(val_losses)

    # Tracer les courbes de perte et d'exactitude
    plt.figure(figsize=(12, 5))

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('LSTM_Loss.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.savefig('LSTM_Accuracy.png')
    plt.show()

    torch.save(model.state_dict(), 'classification.pt')
    print('Trained model saved to classification_custom.pt')
    print('Training complete!')
    #######################################################################
    custom_lightning_module.model.eval()
    test_loss = 0.0
    test_accuracy_values = []
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(custom_lightning_module.device)
            labels = labels.to(custom_lightning_module.device)

            outputs = custom_lightning_module.model(images)
            loss = custom_lightning_module.criterion(outputs, labels)
            test_loss += loss.item()

            accuracy = accuracy_score(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
            test_accuracy_values.append(accuracy)

            all_predicted.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_test_loss = test_loss / len(test_loader)
    average_test_accuracy = np.mean(test_accuracy_values)

    # Calcul de la matrice de confusion
    confusion = confusion_matrix(all_labels, all_predicted)
    # Afficher la matrice de confusion sous forme de figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.title('Matrice de Confusion')
    plt.savefig('Confusion_Matrix.png')
    plt.show()



    # Afficher les résultats
    print(f'Test Loss: {average_test_loss}')
    print(f'Test Accuracy: {average_test_accuracy}')
    print(f'Confusion Matrix:\n{confusion}')

    print(f'Average Test Loss: {average_test_loss}')
    print(f'Average Test Accuracy: {average_test_accuracy}')

