from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml_model import predict_image, predict_video
import os

def home(request):
    return render(request, 'detector/home.html')


def detect(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location='media/uploads/')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join('media/uploads/', filename)

        result = predict_image(file_path)
        result['uploaded_image'] = 'uploads/' + filename

        return render(request, 'detector/result.html', result)

    return render(request, 'detector/home.html')


def detect_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        uploaded_file = request.FILES['video']
        fs = FileSystemStorage(location='media/uploads/')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join('media/uploads/', filename)

        result = predict_video(file_path)

        if 'error' in result:
            return render(request, 'detector/home.html', {'error': result['error']})

        result['uploaded_video'] = 'uploads/' + filename

        return render(request, 'detector/video_result.html', result)

    return render(request, 'detector/home.html')
