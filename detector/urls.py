from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect, name='detect'),
    path('detect-video/', views.detect_video, name='detect_video'),
]
