import mediapipe as mp
import cv2

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils


hands = mphands.Hands(max_num_hands = 1)

video = cv2.VideoCapture(0)


tip = [8,12,16,20]

while True:
    suc,raw = video.read()
    img = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)

    results = hands.process(img)

    lmlist = []
    finger = []
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        hand_label = results.multi_handedness[0].classification[0].label
        for handms in results.multi_hand_landmarks:
            for id,lm in enumerate(handms.landmark):
                
                lmlist.append([id,lm.x,lm.y])
            mpdrawing.draw_landmarks(raw,handms,mphands.HAND_CONNECTIONS,mpdrawing.DrawingSpec(color = (0,0,255),thickness = 1))

    if len(lmlist)>0 and hand_label == 'Right':
        finger.append(lmlist[4][1] < lmlist[4-2][1])
        for i in tip:
            finger.append(lmlist[i][2] < lmlist[i-2][2])
    count = finger.count(True)
    

        
            
    cv2.putText(raw,f"FInger Count :{str(count)}",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    
    cv2.imshow("Finger Count",raw)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

