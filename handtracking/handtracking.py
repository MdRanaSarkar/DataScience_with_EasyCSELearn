import mediapipe as mp
import time
import cv2


mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

vidcap = cv2.VideoCapture(0)
ptime = 0
while True:
    success, img = vidcap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    # print(result)

    if result.multi_hand_landmarks:
        for landmark in result.multi_hand_landmarks:
            for id, lm in enumerate(landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(cx, cy)
                # print(id, lm)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, landmark, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(fps), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
#
