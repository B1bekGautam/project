from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml_model import predict_image
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
