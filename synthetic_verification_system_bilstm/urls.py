from django.contrib import admin
from django.urls import path
from verifier.views import home
from verifier.views import audit

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home, name="home"),
    path("audit/", audit, name="audit"),
]
