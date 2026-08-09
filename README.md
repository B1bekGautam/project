# Deepfake Image and Video Detection System

Django web app detecting real vs AI-generated faces using EfficientNet-B0.

## Setup

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection
```

### 2. Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Model
Download dalle_finetuned_model.pth from Google Drive:

Place in project root folder.

### 4. Create media folder
```bash
mkdir -p media/uploads
```

### 5. Run
```bash
python manage.py runserver
```

Open http://127.0.0.1:8000

