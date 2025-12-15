import mediapipe as mp
import cv2

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils


hands = mphands.Hands(max_num_hands = 1)

video = cv2.VideoCapture(0)

while True:
    suc,raw = video.read()
    img = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handms in results.multi_hand_landmarks:
            mpdrawing.draw_landmarks(raw,handms,mphands.HAND_CONNECTIONS,mpdrawing.DrawingSpec(color = (0,0,255),thickness = 1))
    
    cv2.imshow("Finger Count",raw)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

