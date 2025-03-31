from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import io
import base64

app = Flask(__name__)

# Define the EmotionModel class
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):  # Modify to 7 classes
        super(EmotionModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)  # Load ViT model
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)  # Modify final layer

    def forward(self, x):
        return self.model(x)

# Load the trained model
model = EmotionModel()
model_path = "emotion_model.pth"  # Ensure this file is in the same directory

# Load model with flexible keys
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()  # Set to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Updated Emotion Labels (7 classes)
emotion_labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get image data from request
        data = request.json['image']
        
        # Decode the image from base64
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Transform the image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        predicted_emotion = emotion_labels[predicted_class]

        return jsonify({'prediction': predicted_emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
