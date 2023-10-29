import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1)

labels = ['Oi', 'Tudo Bem', 'Saudade', 'Quanto Tempo']
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Configurações da legenda
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
line_type = 2

while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints is not None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            try:
                imgCrop = img[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
                imgCrop = cv2.resize(imgCrop, (224, 224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                indexVal = np.argmax(prediction)
                text = labels[indexVal]

                # Centralize verticalmente a legenda na parte inferior da imagem
                text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 20  # Posicione a legenda 20 pixels acima da parte inferior

                # Adicione a legenda de filme centralizada na parte inferior
                cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, line_type)

            except:
                continue

        cv2.imshow('Imagem', img)

        # Verifique se a tecla 'q' foi pressionada ou a janela foi fechada
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty('Imagem', 0) == -1:
            break

cap.release()
cv2.destroyAllWindows()
