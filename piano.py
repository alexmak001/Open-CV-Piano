import os
import time

import cv2
import mediapipe as mp
import winsound


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist


cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4,720)
detector = handDetector()

def drawALL(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 75),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    return img


class Button():
    def __init__(self, pos, text, sound, size=[85,85]):
        self.pos = pos
        self.text = text
        self.size = size
        self.sound = sound


path = os.path.dirname(__file__) + "/Sounds/"
sounds = os.listdir("Sounds")

buttonList = []
keys = ["Clap","Bell","Drum","Snare","Air-horn"]
for i, key in enumerate(keys):
    buttonList.append(Button([250*i+50, 100],key, path+sounds[i]))
    print(path+sounds[i])




while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    img = drawALL(img, buttonList)

    if lmlist:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size


            if x < lmlist[8][1] < x+w and y < lmlist[8][2] < y + h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, button.text, (x + 10, y + 75),
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

                winsound.PlaySound(button.sound, winsound.SND_ASYNC)
                time.sleep(0.25)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
