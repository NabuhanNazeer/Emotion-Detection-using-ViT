import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import os
import time

# Step 1: Check CUDA availability
print("🔎 Step 1: Checking CUDA availability...")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Step 2: Check PyTorch & CUDA versions
print("🔎 Step 2: Checking PyTorch & CUDA versions...")  
print(f"✅ PyTorch Version: {torch._version_}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

# Step 3: Set dataset paths
train_data_path = r"C:\Users\Asus\dl\train"  # Update if needed
test_data_path = r"C:\Users\Asus\dl\test"    # Ensure test dataset exists

print(f"✅ Train Dataset path: {train_data_path}")
print(f"✅ Test Dataset path: {test_data_path}")

# Step 4: Verify dataset paths
for path in [train_data_path, test_data_path]:
    if not os.path.exists(path):
        print(f"❌ ERROR: Dataset path {path} does not exist!")
        exit()

# Step 5: Define image transformations
print("🔎 Step 5: Defining image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
print("✅ Transformations set!")

# Step 6: Load datasets
print("🔎 Step 6: Loading datasets...")
start_time = time.time()
try:
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

    print(f"✅ Train: {len(train_dataset)} images in {len(train_dataset.classes)} classes")
    print(f"✅ Test: {len(test_dataset)} images in {len(test_dataset.classes)} classes")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()
print(f"⏳ Dataset loaded in {time.time() - start_time:.2f} seconds")

# Step 7: Create DataLoaders
print("🔎 Step 7: Creating DataLoaders...")
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Total Train Batches: {len(train_loader)}")
print(f"✅ Total Test Batches: {len(test_loader)}")

# Step 8: Load ViT Model
print("🔎 Step 8: Initializing Vision Transformer model...")
try:
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(train_dataset.classes))
    model.to(device)
    print("✅ ViT model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ViT model: {e}")
    exit()

# Step 9: Define loss function and optimizer
print("🔎 Step 9: Setting loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print("✅ Loss function and optimizer set!")

# Step 10: Training loop
num_epochs = 10
print("🔎 Step 10: Starting training loop...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_epoch_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"📝 Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} completed in {time.time() - start_epoch_time:.2f} seconds - Avg Loss: {avg_loss:.4f}")

print("🎉 Training completed successfully!")

# Step 11: Save Model
print("🔎 Step 11: Saving model...")
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "emotion_model.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved at: {save_path}")

# Step 12: Evaluate Accuracy
print("🔎 Step 12: Evaluating Model on Test Data...")

model.eval()  # Set to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Model Accuracy: {accuracy:.2f}% on Test Dataset")
