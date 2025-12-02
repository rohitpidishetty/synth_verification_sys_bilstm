from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import auditor


# Create your views here.
@csrf_exempt
def home(req):
    return HttpResponse(
        "<html><body><center>Welcome to Synthetic Verification System<br/>NFRAC - (Founder) Er. P. Rohit V. Acharya</center></body></html>"
    )


@csrf_exempt
def audit(req):
    if req.method == "POST":
        try:
            data = req.body.decode("utf-8")
            data = json.loads(data)
            tokens = data.get("news", "")
            if tokens.__len__() == 0:
                return JsonResponse({"message": "No tokens received", "status": 300})
            response = auditor(tokens)
            return JsonResponse({"token": response, "status": 200, "q": tokens})
        except Exception as e:
            return JsonResponse({"err": str(e), "status": 300})
    return JsonResponse({"err": "Something went wrong, try again later", "status": 300})
