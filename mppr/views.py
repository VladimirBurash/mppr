import base64
import hashlib
import json
from PIL import Image
from django.http import HttpResponse
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
import dlib
from django.views.decorators.csrf import csrf_exempt
from skimage import io
from scipy.spatial import distance



@method_decorator(csrf_exempt, name='dispatch')
class IndexView(View):

    def post(self, request):
        print(request.body.decode('utf-8'))
        print(json.loads(request.body.decode('utf-8')))
        print(json.loads(request.body.decode('utf-8')).get('data'))
        print('123214')
        data = json.loads(request.body.decode('utf-8')).get('data').split(',')[1]
        data_decoded = base64.b64decode(data)
        image_object = hashlib.sha256(data.encode())
        image_hash = image_object.hexdigest()
        cookie = request.COOKIES.get('user')

        with open('templates/{}.jpg'.format(image_hash), 'wb') as f:
            f.write(data_decoded)
        sp = dlib.shape_predictor('/home/vladimir/models/shape_predictor_68_face_landmarks.dat')
        facerec = dlib.face_recognition_model_v1('/home/vladimir/models/dlib_face_recognition_resnet_model_v1.dat')
        detector = dlib.get_frontal_face_detector()

        img = io.imread('templates/users/{}.jpg'.format(cookie))
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
        face_descriptor1 = facerec.compute_face_descriptor(img, shape)
        img = io.imread('templates/{}.jpg'.format(image_hash))
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
        face_descriptor2 = facerec.compute_face_descriptor(img, shape)

        a = distance.euclidean(face_descriptor1, face_descriptor2)
        if a < 0.6:
            response = {'status': True}
        elif a > 0.6:
            response = {'status': False}
        return JsonResponse(response)


@method_decorator(csrf_exempt, name='dispatch')
class AddView(View):
    def post(self, request):
        data = json.loads(request.body.decode('utf-8')).get('data').split(',')[1]
        data_decoded = base64.b64decode(data)
        image_object = hashlib.sha256(data.encode())
        image_hash = image_object.hexdigest()

        with open('templates/users/{}.jpg'.format(image_hash), 'wb') as f:
            f.write(data_decoded)

        resp = JsonResponse({'status':'ok'})
        resp.set_cookie(key='user', value=image_hash)
        return resp
