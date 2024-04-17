import cv2
import os
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_directory, "Data")
people_list = os.listdir(data_path)
print('Lista de personas:', people_list)

labels = []
faces_data = []
label = 0

for name_dir in people_list:
    person_path = os.path.join(data_path, name_dir)
    print('Leyendo las imágenes')

    for file_name in os.listdir(person_path):
        print('Rostros:', name_dir + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(os.path.join(person_path, file_name), 0))
        # image = cv2.imread(person_path+'/'+file_name,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label += 1

# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(faces_data, np.array(labels))

# Almacenando el modelo obtenido
# face_recognizer.write('modeloEigenFace.xml')
# face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write(os.path.join(current_directory, 'modeloLBPHFace.xml'))
print("Modelo almacenado...")
