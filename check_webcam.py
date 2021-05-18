from os import read
import cv2
import face_recognition as fr
from face_recognition.api import face_distance, face_locations
import numpy as np

from os.path import (
    abspath,
    join,
    dirname,
    splitext,
)
from os import listdir

PWD = dirname(abspath(__file__))
IMGDIR = join(PWD, 'images')
files = listdir(IMGDIR)

def is_image(filename):
    name, ext = splitext(filename)
    if ext == '.jpg' or ext == 'png':
        return True
    
    return False

def capture_frames(video_capture, known_face_encodings, known_face_names, tol=.6):
    while True:
        ret, frame = video_capture.read()
        frame_rgb = frame[:, :, ::-1]
        face_locations = fr.face_locations(frame_rgb)
        face_encodings = fr.face_encodings(frame_rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(
                known_face_encodings=known_face_encodings,
                face_encoding_to_check=face_encoding,
                tolerance=tol,
            )
            name = 'Unknown'
            face_distances = fr.face_distance(
                known_face_encodings,
                face_encoding,
            )
            match_best = np.argmin(face_distances)
            if matches[match_best]:
                name = known_face_names[match_best]
            
            cv2.rectangle(frame, (left, top), (right, bottom), color=(0,0,255), thickness=1)
            cv2.rectangle(frame, (left, bottom-30), (right, bottom), color=(255, 0, 0), thickness=cv2.FILLED)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, color=(0, 255, 0), thickness=1)
        cv2.imshow('Webcam face recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    video_capture = cv2.VideoCapture(0)
    images = [
        file for file in files if is_image(file)
    ]
    face_encodings = []
    face_names = []
    for image in images:
        path_img = join(IMGDIR, image)
        img = fr.load_image_file(path_img)
        face_encodings.append(fr.face_encodings(img)[0])
        face_names.append(splitext(image)[0])

    capture_frames(
        video_capture,
        face_encodings,
        face_names,
    )
