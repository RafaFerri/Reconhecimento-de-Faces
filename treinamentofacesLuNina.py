
import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
from PIL import Image

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join("luenina/treinamento", "*.jpg")):
    imagemFace = Image.open(arquivo).convert('RGB')
    imagem = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        listaDescritorFacial = [df for df in descritorFacial]
        npArrayDescritorFacial = np.array(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        indice[idx] = arquivo
        idx += 1


np.save("recursos/descritores_luenina.npy", descritoresFaciais)
with open("recursos/indices_luenina.pickle", 'wb') as f:
    cPickle.dump(indice, f)

