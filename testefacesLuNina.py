import os
import glob
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices_luenina.pickle")
descritoresFaciais = np.load("recursos/descritores_luenina.npy")
limiar = 0.5

for arquivo in glob.glob(os.path.join("luenina/teste", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem)
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]

        if distanciaMinima <= limiar:
            if os.path.split(indices[minimo])[1].split(".")[0].split("0")[0] == "lu":
                nome = "Luiza"
            else:
                nome = "Maria"
        else:
            nome = ' '

        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 3)
        texto = "{}".format(nome)
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

    cv2.imshow("Detector Lu e Nina", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()