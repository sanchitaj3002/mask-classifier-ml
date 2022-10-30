import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')


results = {0: 'without mask', 1: 'mask'}
Color_interchange = {0: (0, 0, 255), 1: (0, 255, 0)}

rect_size = 4
cam = cv2.VideoCapture(0)

haarcascade = cv2.CascadeClassifier("/Users/atharvajoshi/Documents/mask detection project/haarcascade_frontalface_default.xml")

while True:
    (rval, im) = cam.read() # rval = True or False
    im = cv2.flip(im, 1, 1) # 0 is rotate about x axis, 1 is rotate around y axis

    resized_total_img = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(resized_total_img)

    for f in faces:
        (x, y, w, h) = [j * rect_size for j in f]

        face_img = im[y:y + h, x:x + w] # Cutting image from background
        face_resized = cv2.resize(face_img, (150, 150)) # Resizing image to 150 , 150 
        normalized = face_resized / 255.0 # Dividing every pixel by 255 to reduce size
        reshaped = np.reshape(normalized, (1, 150, 150, 3)) # format = (Number of images, Width, Height, Color Channels)
        reshaped = np.vstack([reshaped]) # Stack faces one upon other
        result = model.predict(reshaped) # Main function to get values (p(un),p(masked))

        label = np.argmax(result, axis=1)[0] # Sort index of max value

        cv2.rectangle(im, (x, y), (x + w, y + h), Color_interchange[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), Color_interchange[label], 1)
        cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('MASK DETECTION REAL TIME', im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cam.release()

cv2.destroyAllWindows()
