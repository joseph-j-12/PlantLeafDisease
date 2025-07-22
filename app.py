import torch
import torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr


# --- 1. Model Definition ---
# The exact same HybridCNNTransformer class from your notebook must be defined here
# so that PyTorch knows how to load the saved model weights.
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNTransformer, self).__init__()
        # CNN Backbone (EfficientNet)
        self.cnn_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool=""
        )
        cnn_feature_dim = self.cnn_backbone.num_features

        # Projection layer to match transformer dimension
        embed_dim = 256
        self.projection = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True, dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classifier Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Pass through CNN
        x = self.cnn_backbone(x)
        # Project to transformer embedding dimension
        x = self.projection(x)
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Pass through transformer
        x = self.transformer_encoder(x)
        # Get CLS token output
        cls_output = x[:, 0]
        # Pass through classifier
        cls_output = self.dropout(cls_output)
        output = self.fc(cls_output)
        return output


# --- 2. Setup and Configuration ---
# Define the same parameters and transformations as in your notebook
class_names = ["Apple_Black_Rot", "Apple_Cedar_Rust", "Apple_Healthy", "Apple_Scab"]
IMG_SIZE = 224
MODEL_PATH = (
    "best_model.pth"  # Make sure your saved model file is in the same directory
)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transformations used for validation/testing
val_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- 3. Load the Model ---
# Initialize the model architecture
model = HybridCNNTransformer(num_classes=len(class_names)).to(device)

# Load the saved weights
# Note: map_location=device ensures the model loads correctly whether you're on a CPU or GPU machine
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Set the model to evaluation mode
model.eval()


# --- 4. Prediction Function ---
# This function takes a PIL image and returns a dictionary of class confidences
def predict(image):
    """
    Takes a PIL image, preprocesses it, and returns a dictionary of
    class names to their predicted probabilities.
    """
    if image is None:
        return None

    # Gradio provides the image as a PIL Image object.
    # We apply the same transformations as our test set.
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Create a dictionary of class names and their confidence scores
    confidences = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return confidences


# --- 5. Create and Launch the Gradio Interface ---
# Define the Gradio interface components
# Input: An image upload box
# Output: A label component to display the predictions
# We also provide a title, description, and some example images for users to try.
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Leaf Image"),
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    title="Plant Leaf Disease Classifier",
    description="Upload an image of an apple leaf to classify its disease. This model can identify Black Rot, Cedar Rust, Scab, or a Healthy leaf.",
    examples=[
        # Add paths to a few example images if you have them.
        # If not, you can remove the 'examples' list.
        # e.g., ['path/to/healthy_leaf.jpg', 'path/to/scab_leaf.jpg']
    ],
)

# Launch the web interface
if __name__ == "__main__":
    print("Launching Gradio interface... Go to the URL below in your browser.")
    iface.launch()
