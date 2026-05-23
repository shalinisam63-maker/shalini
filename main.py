import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import copy
import random
from collections import Counter

# -------------------------------
# STEP 1: Data Augmentation
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('data/chest_xray/train', transform=transform)
test_dataset = datasets.ImageFolder('data/chest_xray/test', transform=transform)

# -------------------------------
# STEP 2: Reduce Dataset
# -------------------------------
def get_subset(dataset, size=2000):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices[:size])

train_dataset = get_subset(train_dataset, 2000)
test_dataset = get_subset(test_dataset, 500)

# -------------------------------
# STEP 3: Show Sample X-rays (POP-UP)
# -------------------------------
def show_images(dataset):
    fig, axes = plt.subplots(1,5, figsize=(10,3))
    for i in range(5):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title("PNEUMONIA" if label==1 else "NORMAL")
        axes[i].axis('off')
    plt.show()

show_images(train_dataset)

# -------------------------------
# STEP 4: Split into clients
# -------------------------------
def split_clients(dataset, num_clients=3):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_size = len(dataset) // num_clients
    return [Subset(dataset, indices[i*split_size:(i+1)*split_size]) for i in range(num_clients)]

clients = split_clients(train_dataset, 3)

# -------------------------------
# STEP 5: Model
# -------------------------------
def get_model():
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# -------------------------------
# STEP 6: Weighted Loss
# -------------------------------
labels = [label for _, label in train_dataset]
class_counts = Counter(labels)
weights = [1.0/class_counts[i] for i in range(len(class_counts))]
class_weights = torch.tensor(weights)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# -------------------------------
# STEP 7: Training (2 epochs)
# -------------------------------
train_losses = []

def train(model, loader, epochs=4):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        total_loss = 0
        for images, labels in loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(loader))
    return model.state_dict()

# -------------------------------
# STEP 8: Federated Averaging
# -------------------------------
def federated_avg(weights, sizes):
    avg = copy.deepcopy(weights[0])
    for key in avg:
        avg[key] = sum(weights[i][key]*sizes[i] for i in range(len(weights))) / sum(sizes)
    return avg

# -------------------------------
# STEP 9: Federated Training (4 rounds)
# -------------------------------
global_model = get_model()
num_rounds = 6

for round in range(num_rounds):
    local_weights = []
    sizes = []

    for client_data in clients:
        loader = DataLoader(client_data, batch_size=32, shuffle=True)
        local_model = get_model()
        local_model.load_state_dict(global_model.state_dict())

        weights = train(local_model, loader)
        local_weights.append(weights)
        sizes.append(len(client_data))

    global_weights = federated_avg(local_weights, sizes)
    global_model.load_state_dict(global_weights)

    print(f"Round {round+1} done")

# -------------------------------
# STEP 10: Testing
# -------------------------------
def test(model, dataset):
    loader = DataLoader(dataset, batch_size=32)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs,1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL","PNEUMONIA"]))

    return all_labels, all_preds

labels, preds = test(global_model, test_dataset)

# -------------------------------
# STEP 11: Training Loss Graph (POP-UP)
# -------------------------------
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# -------------------------------
# STEP 12: Confusion Matrix (POP-UP)
# -------------------------------
cm = confusion_matrix(labels, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["NORMAL","PNEUMONIA"], yticklabels=["NORMAL","PNEUMONIA"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# STEP 13: Save Model
# -------------------------------
torch.save(global_model.state_dict(), "federated_model.pth")

# -------------------------------
# STEP 14: Prediction Function
# -------------------------------
def predict_image(model, img_path):
    img = Image.open(img_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output,1)

    return "PNEUMONIA" if pred.item()==1 else "NORMAL"
print(predict_image(global_model, r"C:\Users\shalini s\Downloads\DL_Project\data\chest_xray\test\PNEUMONIA\person94_bacteria_457.jpeg"))
import cv2
import numpy as np

def grad_cam(model, img_path):
    model.eval()

    # Load image
    img = Image.open(img_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Hook for gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Target layer (last conv layer of ResNet)
    target_layer = model.layer4[1].conv2

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients and activations
    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    # Compute weights
    weights = np.mean(grads, axis=(1,2))

    # Weighted combination
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Resize to image size
    cam = cv2.resize(cam, (128,128))

    # Convert original image to array
    img_np = np.array(img.resize((128,128)))

    # Apply heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + np.stack([img_np]*3, axis=-1)

    # Plot
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Original X-ray")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(superimposed.astype(np.uint8))
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.show()

    handle_fw.remove()
    handle_bw.remove()

grad_cam(global_model, r"C:\Users\shalini s\Downloads\DL_Project\data\chest_xray\test\PNEUMONIA\person94_bacteria_457.jpeg")
# Example:
# print(predict_image(global_model, "data/chest_xray/sample1.jpeg"))