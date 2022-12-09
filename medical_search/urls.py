from django.urls import path
from . import views

app_name = 'medical_search'

urlpatterns = [
    path('', views.search, name='search'),
    path('<path1>/<path2>/<path3>', views.view_content, name='content')
]