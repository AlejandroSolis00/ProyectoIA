import cv2
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_directory, "Data")
image_paths = os.listdir(data_path)
print('image_paths =', image_paths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(os.path.join(current_directory, 'modeloLBPHFace.xml'))

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('Desconocido.mp4')

face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Definir el tamaño de la ventana de resultados
window_width = 854
window_height = 480

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = gray.copy()

    faces = face_classif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 55:
            cv2.putText(frame, '{}'.format(image_paths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Redimensionar el marco antes de mostrarlo
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
