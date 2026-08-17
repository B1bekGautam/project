import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

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

# MTCNN for face detection (used in video pipeline)
mtcnn = MTCNN(keep_all=False, device=DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ['Fake', 'Real']


def predict_tensor(input_tensor, img_array, output_filename):
    """
    Shared core prediction function used by both image and video pipelines.
    input_tensor: preprocessed torch tensor (1, 3, 224, 224) on DEVICE
    img_array:    numpy float32 array (224, 224, 3) in [0, 1] for Grad-CAM overlay
    output_filename: full path where the Grad-CAM heatmap will be saved
    Returns: (verdict, confidence, is_fake)
    """
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probs[0][pred_class].item() * 100

    verdict = CLASS_NAMES[pred_class]
    is_fake = pred_class == 0

    # Grad-CAM
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    plt.imsave(output_filename, heatmap)

    return verdict, round(confidence, 2), is_fake


def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    heatmap_filename = 'gradcam_' + os.path.basename(image_path)
    heatmap_path = os.path.join(BASE_DIR, 'media', 'uploads', heatmap_filename)

    verdict, confidence, is_fake = predict_tensor(input_tensor, img_array, heatmap_path)

    print(f"Image verdict: {verdict} | Confidence: {confidence:.2f}%")

    return {
        'verdict': verdict,
        'confidence': confidence,
        'heatmap': 'uploads/' + heatmap_filename,
        'is_fake': is_fake
    }


def predict_video(video_path, fps_sample=2):
    """
    Video deepfake detection pipeline:
    1. Extract frames at fps_sample rate
    2. Detect face in each frame using MTCNN
    3. Classify each face using the same model
    4. Aggregate frame-level scores via weighted majority voting
    5. Generate Grad-CAM on the most confident frame
    Returns result dict for the view.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video file.'}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25  # fallback
    frame_interval = max(1, int(video_fps / fps_sample))

    frame_results = []   # list of (pred_class, confidence, input_tensor, img_array)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            # Face detection
            boxes, _ = mtcnn.detect(pil_frame)

            if boxes is not None and len(boxes) > 0:
                # Use the first detected face
                box = boxes[0]
                x1, y1, x2, y2 = [max(0, int(c)) for c in box]
                face_crop = pil_frame.crop((x1, y1, x2, y2))
            else:
                # No face detected — use full frame resized
                face_crop = pil_frame

            face_resized = face_crop.resize((224, 224))
            img_array = np.array(face_resized) / 255.0
            input_tensor = transform(face_resized).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = output.argmax(1).item()
                confidence = probs[0][pred_class].item() * 100

            frame_results.append((pred_class, confidence, input_tensor, img_array))
            print(f"Frame {frame_index}: {CLASS_NAMES[pred_class]} ({confidence:.2f}%)")

        frame_index += 1

    cap.release()

    if not frame_results:
        return {'error': 'No frames could be processed from the video.'}

    # --- Weighted majority voting ---
    # Each frame votes with its confidence as weight
    fake_weight = sum(conf for cls, conf, _, _ in frame_results if cls == 0)
    real_weight = sum(conf for cls, conf, _, _ in frame_results if cls == 1)
    total_weight = fake_weight + real_weight

    fake_score = (fake_weight / total_weight) * 100
    real_score = (real_weight / total_weight) * 100

    if fake_weight >= real_weight:
        final_verdict = 'Fake'
        final_confidence = round(fake_score, 2)
        is_fake = True
        target_class = 0
    else:
        final_verdict = 'Real'
        final_confidence = round(real_score, 2)
        is_fake = False
        target_class = 1

    # --- Grad-CAM on most confident frame matching final verdict ---
    best_frame = max(
        [f for f in frame_results if f[0] == target_class],
        key=lambda f: f[1]
    )
    _, _, best_tensor, best_img_array = best_frame

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    heatmap_filename = f'gradcam_video_{video_basename}.png'
    heatmap_path = os.path.join(BASE_DIR, 'media', 'uploads', heatmap_filename)

    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=best_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(
        best_img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    plt.imsave(heatmap_path, heatmap)

    total_frames = len(frame_results)
    fake_frames = sum(1 for cls, _, _, _ in frame_results if cls == 0)
    real_frames = total_frames - fake_frames

    print(f"Video verdict: {final_verdict} | Confidence: {final_confidence}% | "
          f"Frames: {total_frames} (Fake: {fake_frames}, Real: {real_frames})")

    return {
        'verdict': final_verdict,
        'confidence': final_confidence,
        'heatmap': 'uploads/' + heatmap_filename,
        'is_fake': is_fake,
        'total_frames': total_frames,
        'fake_frames': fake_frames,
        'real_frames': real_frames,
        'fake_score': round(fake_score, 2),
        'real_score': round(real_score, 2),
    }
