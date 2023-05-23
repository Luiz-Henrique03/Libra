import cv2
import numpy as np
import mediapipe as mp
import pickle

cap = cv2.VideoCapture(0)
model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0:'A',1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'I', 8:'L',9:'M', 10:'N',11:'O', 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T',17:'U', 18:'V'}

while True:
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        print(predicted_character)



    cv2.imshow('frame',frame)
    cv2.waitKey(25)
cap.release()
cv2.destroyAllWindows()