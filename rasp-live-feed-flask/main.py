import cv2
import sys
from flask import Flask, render_template, Response
from webcamvideostream import WebcamVideoStream
from flask_basicauth import BasicAuth
import time
import threading

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'pi'
app.config['BASIC_AUTH_PASSWORD'] = 'pi'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)
last_epoch = 0

# Load local Haar Cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def gen(camera):
    while True:
        if camera.stopped:
            break
        frame = camera.read()
        frame_with_faces = detect_faces(frame)
        ret, jpeg = cv2.imencode('.jpg', frame_with_faces)
        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            print("frame is none")

@app.route('/video_feed')
def video_feed():
    return Response(gen(WebcamVideoStream().start()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
