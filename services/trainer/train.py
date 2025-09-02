import os
import sys
import time
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import json
import tempfile

# Environment variables
DATA_ROOT = os.getenv("DATA_ROOT", "/data")
EXPERIMENT = os.getenv("EXPERIMENT", "ivis-resnet")
MODEL_NAME = os.getenv("MODEL_NAME", "ivis_resnet")
EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH", "32"))
IMG_SIZE = int(os.getenv("IMG", "224"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_data_loaders():
    """Setup data loaders for train and validation."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    
    for phase in ['train', 'val']:
        phase_dir = os.path.join(DATA_ROOT, phase)
        if not os.path.exists(phase_dir):
            print(f"Warning: {phase_dir} does not exist. Creating dummy data...")
            os.makedirs(phase_dir, exist_ok=True)
            # Create dummy class directories
            for class_name in ['class1', 'class2']:
                os.makedirs(os.path.join(phase_dir, class_name), exist_ok=True)
        
        try:
            image_datasets[phase] = datasets.ImageFolder(phase_dir, data_transforms[phase])
            dataloaders[phase] = DataLoader(
                image_datasets[phase], 
                batch_size=BATCH_SIZE,
                shuffle=(phase == 'train'),
                num_workers=2
            )
            dataset_sizes[phase] = len(image_datasets[phase])
        except Exception as e:
            print(f"Error loading {phase} data: {e}")
            sys.exit(1)
    
    if len(image_datasets['train'].classes) < 2:
        print("Error: Need at least 2 classes. Please add images to data/train/<class> folders.")
        sys.exit(1)
    
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    """Train the model."""
    since = time.time()
    best_acc = 0.0
    best_model_state = None
    
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            with tqdm(dataloaders[phase], desc=f'{phase.capitalize()}') as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    pbar.set_postfix({'loss': loss.item()})
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Log metrics to MLflow
            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_acc", epoch_acc.item(), step=epoch)
            
            # Store metrics
            metrics_history[f'{phase}_loss'].append(epoch_loss)
            metrics_history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state = model.state_dict()
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Log best accuracy
    mlflow.log_metric("best_val_acc", best_acc.item())
    
    return model

def main():
    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT)
    
    # Load data
    print("Loading data...")
    dataloaders, dataset_sizes, class_names = setup_data_loaders()
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("image_size", IMG_SIZE)
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("device", str(DEVICE))
	#log class names as artifacts
	with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="metadata")
        
        # Initialize model
        print("Initializing model...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model = model.to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train model
        print("Starting training...")
        model = train_model(
            model, criterion, optimizer, exp_lr_scheduler,
            dataloaders, dataset_sizes, num_epochs=EPOCHS
        )
        
        # Log model
        print("Logging model to MLflow...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=mlflow.models.infer_signature(
                torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE),
                model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE))
            ),
            extra_pip_requirements=["torch==2.3.1", "torchvision==0.18.1", "pillow==10.3.0"]
        )
        
        # Transition model to production
        client = mlflow.MlflowClient()
        
        # Wait for model to be registered
        import time
        for _ in range(30):
            try:
                latest_version = client.get_latest_versions(MODEL_NAME)
                if latest_version:
                    version = latest_version[0].version
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=version,
                        stage="Production"
                    )
                    print(f"Model {MODEL_NAME} version {version} transitioned to Production")
                    break
            except:
                time.sleep(1)
        
        print("Training completed successfully!")

if __name__ == "__main__":
    main()
