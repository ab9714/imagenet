import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import json

with open('imagenet_classes.json') as f:
    labels = json.load(f)

# Load pre-trained ResNet50 with new weights parameter
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    # Preprocess the image
    img_tensor = preprocess(Image.fromarray(image))
    img_tensor = img_tensor.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Create result dictionary
    results = {labels[str(idx.item())]: prob.item() for prob, idx in zip(top5_prob, top5_idx)}
    
    return results

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet50 ImageNet Classifier",
    description="Upload an image to classify it using ResNet50 trained on ImageNet"
)

if __name__ == "__main__":
    iface.launch()