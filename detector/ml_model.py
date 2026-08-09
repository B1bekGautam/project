import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'dalle_finetuned_model.pth')

print(f"Loading model from: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ['Fake', 'Real']

def predict_image(image_path):
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Image path: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probs[0][pred_class].item() * 100

    # Debug print
    print(f"Raw output: {output}")
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {pred_class} ({CLASS_NAMES[pred_class]})")
    print(f"Confidence: {confidence:.2f}%")

    verdict = CLASS_NAMES[pred_class]

    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    heatmap_filename = 'gradcam_' + os.path.basename(image_path)
    heatmap_path = os.path.join(BASE_DIR, 'media', 'uploads', heatmap_filename)
    plt.imsave(heatmap_path, heatmap)

    return {
        'verdict': verdict,
        'confidence': round(confidence, 2),
        'heatmap': 'uploads/' + heatmap_filename,
        'is_fake': pred_class == 0
    }