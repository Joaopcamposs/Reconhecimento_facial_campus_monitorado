import cv2
import os
import numpy as np


def trainLBPH():
    lbph = cv2.face.LBPHFaceRecognizer_create()

    def getImageAndId():
        paths = [os.path.join("pictures", f) for f in os.listdir("pictures")]
        faces = []
        ids = []

        for imagePath in paths:
            face_image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            ids.append(id)
            faces.append(face_image)
        return np.array(ids), faces

    ids, faces = getImageAndId()

    print("Training...")

    lbph.train(faces, ids)
    lbph.write("classifierLBPH.yml")

    print("Training completed!")
