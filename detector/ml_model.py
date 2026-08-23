import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from facenet_pytorch import MTCNN
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'dalle_finetuned_model.pth')

print(f"Loading model from: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

# Load EfficientNet model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on: {DEVICE}")

# Load MTCNN for face detection
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=DEVICE
)
print("MTCNN loaded")

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ['Fake', 'Real']

def predict_image(image_path):
    print(f"Processing: {image_path}")

    # Load original image
    img = Image.open(image_path).convert('RGB')

    # Step 1 — MTCNN face detection
    face_tensor = mtcnn(img)

    if face_tensor is None:
        print("No face detected by MTCNN")
        return {
            'verdict': 'No Face Detected',
            'confidence': 0,
            'heatmap': None,
            'is_fake': False,
            'no_face': True
        }

    print("Face detected by MTCNN")

    # Convert MTCNN output back to PIL for Grad-CAM visualization
    # MTCNN returns tensor in range [-1, 1] — convert to [0, 1] for display
    face_display = face_tensor.permute(1, 2, 0).numpy()
    face_display = (face_display - face_display.min()) / (face_display.max() - face_display.min())
    face_display = np.clip(face_display, 0, 1)

    # Step 2 — Prepare input tensor for EfficientNet
    # MTCNN already resized to 224x224 and normalized to [-1,1]
    # We need to renormalize to ImageNet normalization
    face_pil = Image.fromarray((face_display * 255).astype(np.uint8))
    input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

    # Step 3 — Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probs[0][pred_class].item() * 100

    print(f"Predicted: {CLASS_NAMES[pred_class]} ({confidence:.2f}%)")

    verdict = CLASS_NAMES[pred_class]

    # Step 4 — Grad-CAM on face crop
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(
        face_display.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # Save heatmap
    heatmap_filename = 'gradcam_' + os.path.basename(image_path)
    heatmap_path = os.path.join(BASE_DIR, 'media', 'uploads', heatmap_filename)
    plt.imsave(heatmap_path, heatmap)

    # Save face crop for display
    face_filename = 'face_' + os.path.basename(image_path)
    face_path = os.path.join(BASE_DIR, 'media', 'uploads', face_filename)
    face_pil.save(face_path)

    return {
        'verdict': verdict,
        'confidence': round(confidence, 2),
        'heatmap': 'uploads/' + heatmap_filename,
        'face_crop': 'uploads/' + face_filename,
        'is_fake': pred_class == 0,
        'no_face': False
    }



import cv2

def predict_video(video_path, fps_sample=2):
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video file'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / fps_sample))

    frame_scores = []
    frame_count = 0
    processed_frames = 0
    gradcam_saved = False
    heatmap_filename = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert frame to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # MTCNN face detection
            face_tensor = mtcnn(pil_img)

            if face_tensor is not None:
                # Convert for display
                face_display = face_tensor.permute(1, 2, 0).numpy()
                face_display = (face_display - face_display.min()) / (face_display.max() - face_display.min())
                face_display = np.clip(face_display, 0, 1)

                # Prepare input tensor
                face_pil = Image.fromarray((face_display * 255).astype(np.uint8))
                input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_class = output.argmax(1).item()
                    confidence = probs[0][pred_class].item()

                frame_scores.append({
                    'pred_class': pred_class,
                    'confidence': confidence
                })

                # Save Grad-CAM for first detected face only
                if not gradcam_saved:
                    target_layers = [model.conv_head]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(pred_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                    heatmap = show_cam_on_image(
                        face_display.astype(np.float32),
                        grayscale_cam,
                        use_rgb=True
                    )
                    heatmap_filename = 'gradcam_video_' + os.path.basename(video_path) + '.png'
                    heatmap_path = os.path.join(BASE_DIR, 'media', 'uploads', heatmap_filename)
                    plt.imsave(heatmap_path, heatmap)
                    gradcam_saved = True

                processed_frames += 1

        frame_count += 1

    cap.release()

    if not frame_scores:
        return {'error': 'No faces detected in video'}

    # Temporal aggregation — weighted majority voting
    fake_score = sum(s['confidence'] for s in frame_scores if s['pred_class'] == 0)
    real_score = sum(s['confidence'] for s in frame_scores if s['pred_class'] == 1)
    total_frames = len(frame_scores)
    fake_frames = sum(1 for s in frame_scores if s['pred_class'] == 0)
    real_frames = total_frames - fake_frames

    if fake_score > real_score:
        verdict = 'Fake'
        confidence = round((fake_score / (fake_score + real_score)) * 100, 2)
        is_fake = True
    else:
        verdict = 'Real'
        confidence = round((real_score / (fake_score + real_score)) * 100, 2)
        is_fake = False

    print(f"Video verdict: {verdict} ({confidence}%)")
    print(f"Frames analyzed: {total_frames} | Fake: {fake_frames} | Real: {real_frames}")

    total_score = fake_score + real_score
    fake_pct = round((fake_score / total_score) * 100, 1) if total_score > 0 else 0
    real_pct = round((real_score / total_score) * 100, 1) if total_score > 0 else 0

    return {
        'verdict': verdict,
        'confidence': confidence,
        'is_fake': is_fake,
        'total_frames': total_frames,
        'fake_frames': fake_frames,
        'real_frames': real_frames,
        'fake_score': fake_pct,
        'real_score': real_pct,
        'heatmap': 'uploads/' + heatmap_filename if heatmap_filename else None,
        'no_face': False
    }