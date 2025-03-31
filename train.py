import numpy as np
import matplotlib.pyplot as plt
import cv2

class RBM:
    def _init_(self, n_visible, n_hidden):
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        self.hidden_bias = np.random.randn(n_hidden) * 0.1
        self.visible_bias = np.random.randn(n_visible) * 0.1

    def sample_hidden(self, visible):
        activation = np.dot(visible, self.weights) + self.hidden_bias
        probabilities = 1 / (1 + np.exp(-activation))
        return np.random.binomial(1, probabilities)

    def sample_visible(self, hidden):
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        probabilities = 1 / (1 + np.exp(-activation))
        return np.random.binomial(1, probabilities)

    def train(self, data, learning_rate, epochs):
        for epoch in range(epochs):
            v0 = data
            h0 = self.sample_hidden(v0)
            v1 = self.sample_visible(h0)
            h1 = self.sample_hidden(v1)
            self.weights += learning_rate * (np.dot(v0.T, h0) - np.dot(v1.T, h1))
            self.visible_bias += learning_rate * np.mean(v0 - v1, axis=0)
            self.hidden_bias += learning_rate * np.mean(h0 - h1, axis=0)

class DBM:
    def _init_(self, layer_sizes):
        self.rbms = [RBM(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def pretrain_layers(self, data, learning_rate, epochs):
        for i, rbm in enumerate(self.rbms):
            print(f"Pretraining RBM Layer {i+1}/{len(self.rbms)}")
            rbm.train(data, learning_rate, epochs)
            data = rbm.sample_hidden(data)

    def finetune(self, data, learning_rate, epochs):
        for epoch in range(epochs):
            up_data = data
            up_pass_data = [data]
            for rbm in self.rbms:
                up_data = rbm.sample_hidden(up_data)
                up_pass_data.append(up_data)

            down_data = up_data
            for i, rbm in enumerate(reversed(self.rbms)):
                down_data = rbm.sample_visible(down_data)
                if i < len(self.rbms) - 1:
                    self.rbms[-i-1].train(up_pass_data[-i-2], learning_rate, 1)
            print(f"Finetuning Epoch {epoch+1}/{epochs}")

    def forward_pass(self, visible):
        hidden_data = visible
        for rbm in self.rbms:
            hidden_data = rbm.sample_hidden(hidden_data)
        return hidden_data

# Load and preprocess image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_resized = cv2.resize(image, (224, 224))
    image_data = image_resized.flatten() / 255.0
    return image_data, image_resized

# Visualize detected objects
def visualize_results(image, output):
    objects = ['Tree', 'Building', 'Person']
    detected_objects = [objects[i] for i in range(3) if output[0, i] == 1]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected: {', '.join(detected_objects) if detected_objects else 'None'}")
    plt.axis('off')
    plt.show()

# Initialize and train the DBM
image_path = '/mnt/data/image.png'
image_data, original_image = load_image(image_path)
dbm = DBM([224*224*3, 512, 256, 3])
dbm.pretrain_layers(image_data.reshape(1, -1), learning_rate=0.01, epochs=50)
dbm.finetune(image_data.reshape(1, -1), learning_rate=0.01, epochs=10)

# Get the output prediction
output = dbm.forward_pass(image_data.reshape(1, -1))
print("Output from DBM:", output)
visualize_results(original_image, output)