import cv2
import os
from threading import Thread
import time
import numpy as np

class WebcamVideoStream:
    def __init__(self, src=0):
        print("init")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_cascade = cv2.CascadeClassifier(os.path.join(current_dir, 'haarcascade_frontalface_default.xml'))
        self.stream = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
        self.stopped = False
        self.frame = None  # Initialiser self.frame
        time.sleep(2.0)
    
    def start(self):
        print("start thread")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        print("read")
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.frame = frame
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
