import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import time
import pandas as pd
from utils import load_vgg16, load_inception, load_resnet, create_logger, create_loaders
import shutil

def reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

models = {
    'vgg16': load_vgg16(),
    'resnet': load_resnet(),
    'inception': load_inception()
}


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def evaluate_model(model, val_data, device, conf_mat=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loaded = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        for i, (inputs, labels) in enumerate(val_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            if conf_mat:
                y_pred.extend(preds.detach().cpu().numpy())  # Save Prediction
                y_true.extend(labels.detach().cpu().numpy())  # Save Truth
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).float().sum()
            total_loaded += len(labels)

    epoch_loss = running_loss / total_loaded
    epoch_acc = running_corrects / total_loaded * 100.0

    if conf_mat:
        return epoch_loss, epoch_acc, y_true, y_pred

    return epoch_loss, epoch_acc


def train(
    model, train_data, val_data, test_data, num_epochs, device, fold, logger, models_dir
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    stopper = EarlyStopper(patience=10, min_delta=5)

    start_time = time.time()  # (for showing time)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):  # (loop for every epoch)
        logger.info(f"Epoch {epoch} running")  # (printing message)
        """ Training Phase """
        model.train()  # (training model)
        running_loss = 0.0  # (set loss 0)
        running_corrects = 0
        # load a batch data of images
        total_loaded = 0
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward inputs and get output
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).float().sum()
            total_loaded += len(labels)

        epoch_loss = running_loss / total_loaded
        epoch_acc = running_corrects / total_loaded * 100.0
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.detach().cpu().float().item())
        logger.info(
            f"[Train #{epoch} k={fold}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Time: {time.time() - start_time:.4f}s"
        )

        ## Validation step
        start_time = time.time()
        epoch_val_loss, epoch_val_acc = evaluate_model(model, val_data, device)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.detach().cpu().float().item())
        logger.info(
            f"[Val #{epoch} k={fold}] Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}% Time: {time.time() - start_time:.4f}s"
        )
        if stopper.early_stop(epoch_val_loss):
            logger.info(f'the stopper criterion has not been met, stopping training after {epoch} epochs.')
            break

    # Test step
    start_time = time.time()
    test_loss, test_acc, y_pred, y_true = evaluate_model(model, test_data, device, True)
    logger.info(
        f"[Test #{epoch} k={fold}] Loss: {test_loss:.4f} Acc: {test_acc:.4f}% Time: {time.time() - start_time:.4f}s"
    )

    save_path = (
        models_dir / f"custom-classifier-acc{int(epoch_acc)}-loss{int(epoch_loss)}.pth"
    )
    torch.save(model.state_dict(), save_path)

    return (
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        test_loss,
        test_acc,
        y_pred,
        y_true,
    )


def run(
    dataset_path: Path,
    epochs: int,
    num_folds: int,
    batch_size: int,
    logs_dir: Path,
    figs_dir: Path,
    models_dir: Path,
    ):
    """Run KFold training

    Args:
        dataset_path (Path): Path to image folder
        epochs (int): Number of epochs to train in each fold
        num_folds (int): Number of fold partition to be applied to dataset
        batch_size (int): Number of images to load in each step
        logs_dir (Path): Path to save the produced logs
        figs_dir (Path): Path to save the produced figures
        models_dir (Path): Path to save the produced models
    """
    logger = create_logger(logs_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'TRAINING WITH DEVICE: {device}')
    applied_transforms = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # must same as here
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    for key in models:
        if key == 'inception':
            applied_transforms = transforms.Compose(
            [
                transforms.Resize((299, 299)),  # must same as here
                transforms.RandomResizedCrop(size=(224, 224), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.CenterCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        logger.info(f'*** TRAINING MODEL: {key} ***')
        logs_dir_model = logs_dir / key
        models_dir_model = models_dir / key
        figs_dir_model = figs_dir / key
        os.makedirs(logs_dir_model, exist_ok=True)
        os.makedirs(models_dir_model, exist_ok=True)
        os.makedirs(figs_dir_model, exist_ok=True)
        
        dataset = datasets.ImageFolder(dataset_path, applied_transforms)
        kfold = KFold(n_splits=num_folds, shuffle=True)
        logger.info(f"Train dataset size: {len(dataset)}")
        class_names = dataset.classes
        logger.info(f"Class names: {class_names}")

        model = models[key]
        
        if key == 'inception':
            model.aux_logits = False
            
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        test_losses, test_accs = [], []
        y_preds, y_trues = [], []
        train_df = pd.DataFrame(columns=['Fold', 'Epoch', 'AccTrain', 'LossTrain', 'AccVal', 'LossVal'])
        test_df = pd.DataFrame(columns=['Fold', 'Acc', 'Loss'])
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
            logger.info(f'*** RUNNING FOLD {fold} ***')
            train_loader, val_loader, test_loader = create_loaders(batch_size, dataset, train_idx, test_idx)
            model.apply(reset_weights)
            trl, tra, vl, va, test_loss, test_acc, y_pred, y_true = train(
                model,
                train_loader,
                val_loader,
                test_loader,
                epochs,
                device,
                fold,
                logger,
                models_dir_model,
                )
            df_data = {
                'Fold': [fold for i in range(len(trl))],
                'Epoch': [i for i in range(len(trl))],
                'AccTrain': tra,
                'LossTrain': trl,
                'AccVal': va,
                'LossVal': vl
            }
            df = pd.DataFrame(df_data)
            train_df = pd.concat([train_df, df])
            y_preds.extend(y_pred)
            y_trues.extend(y_true)
            train_losses.append(trl)
            train_accs.append(tra)
            val_losses.append(vl)
            val_accs.append(va)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_df.loc[len(test_df)] = [fold, test_acc.cpu().float().item(), test_loss]
        
        cm_ = confusion_matrix(y_preds, y_trues)
        cm = cm_.astype("float") / cm_.sum(axis=1)[:, np.newaxis]
        if cm.shape != (2,2):
            # logger.debug(f'Error on confusion matrix, original is: \n{cm_}')
            # logger.debug(f'calculated matrix is: \n{cm}')
            # logger.debug(f'y_trues={y_trues}')
            # logger.debug(f'y_preds={y_preds}')
            cm = cm[:2, :2]

        df_cm = pd.DataFrame(
            cm,
            index=[i for i in class_names],
            columns=[i for i in class_names],
        )
        df_cm.to_csv(figs_dir_model / "cf_mat.csv", index=False)
        train_df.to_csv(figs_dir_model / "train_metrics.csv", index=False)
        test_df.to_csv(figs_dir_model / "test_metrics.csv", index=False)
        
        train_loss_hist = [l[-1] for l in train_losses]
        train_acc_hist = [l[-1] for l in train_accs]
        val_loss_hist = [l[-1] for l in val_losses]
        val_acc_hist = [l[-1] for l in val_accs]

        train_loss_msg = (
            f"_Train Loss:_ {np.mean(train_loss_hist): .2f}, {np.std(train_loss_hist): .2f}"
        )
        val_loss_msg = (
            f"_Val Loss:_ {np.mean(val_loss_hist): .2f}, {np.std(val_loss_hist): .2f}"
        )
        train_acc_msg = (
            f"_Train Acc:_ {np.mean(train_acc_hist): .2f}, {np.std(train_acc_hist): .2f}"
        )
        val_acc_msg = (
            f"_Val Acc:_ {np.mean(val_acc_hist): .2f}, {np.std(val_acc_hist): .2f}"
        )
        test_loss_msg = (
            f"_Test Loss:_ {np.mean(test_losses): .2f}, {np.std(test_losses): .2f}"
        )
        test_acc_msg = f"_Test Acc:_ {np.mean(val_accs): .2f}, {np.std(val_accs): .2f}"

        logger.info(train_loss_msg.replace("_", ""))
        logger.info(train_acc_msg.replace("_", ""))
        logger.info(val_loss_msg.replace("_", ""))
        logger.info(val_acc_msg.replace("_", ""))
        logger.info(test_loss_msg.replace("_", ""))
        logger.info(test_acc_msg.replace("_", ""))
        
        
    

def parse_arguments() -> argparse.Namespace:
    """Command line arguments parser

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Train simple model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=Path,
        help="Path to the images folder for training the model",
    )

    # Training options
    parser.add_argument(
        "--epochs", action="store", type=int, default=100, required=False
    )
    parser.add_argument(
        "--num-folds", action="store", type=int, default=5, required=False
    )
    parser.add_argument(
        "--batch-size", action="store", type=int, default=32, required=False
    )

    # Logging options
    parser.add_argument(
        "--logs-dir", action="store", type=Path, default=Path("logs"), required=False
    )
    parser.add_argument(
        "--figs-dir", action="store", type=Path, default=Path("figures"), required=False
    )
    parser.add_argument(
        "--models-dir",
        action="store",
        type=Path,
        default=Path("models"),
        required=False,
    )

    parsed_args = parser.parse_args()
    return parsed_args


def main() -> None:
    run(**vars(parse_arguments()))


if __name__ == "__main__":
    main()
