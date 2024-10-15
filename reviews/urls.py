# reviews/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.review_input, name='review_input'),
]