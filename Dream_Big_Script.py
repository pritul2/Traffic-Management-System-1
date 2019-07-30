import cv2
import numpy as np
from sklearn.externals import joblib
from threading import *
import time

refIm = cv2.imread('refFrame.jpg')
refIm2 = cv2.cvtColor(refIm, cv2.COLOR_BGR2GRAY)
roi = np.ones(refIm2.shape, "uint8")
# setting roi
cv2.rectangle(roi, (62, 60), (242, 180), 255, -1)

bg = refIm2.copy()
bg = cv2.bitwise_and(bg, roi)

model = joblib.load("model.cpickle")
TIME = 0


class SetTimer(Thread):
    def run(self):
        canvas = np.zeros((232, 302), "uint8")
        final_time=TIME
        print("predicted time is ",final_time)
        for i in range(final_time):
            #cv2.putText(canvas, str(final_time - i), (int(canvas.shape[0] / 2), int(canvas.shape[1] / 2)),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            time.sleep(1)
            #canvas = np.zeros((232, 302), "uint8")
            #cv2.imshow("canvas",canvas)
            #cv2.waitKey(1)
            print(final_time-i)


def calcFrame(x, y):
    frame_time = int((x * 60 + y) * 35)
    return frame_time


def process(frame):
    vidClone = frame.copy()
    global roi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, roi)
    diff = cv2.absdiff(bg.astype('uint8'), gray)
    # threshold logic#
    thresh = 53
    thresholded = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
    # Opening logic#
    k = 3
    kernel = np.ones((k, k), "uint8")
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening', opening)
    # dilation logic#
    dilate = 15
    dilated = cv2.dilate(opening, None, iterations=dilate)
    # change to _,contour,_ for latest version#
    contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Finding the area of each contour#
    for i in range(len(contour)):

        M = cv2.moments(contour[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour[i])
        if area >= 3700:
            cv2.drawContours(vidClone, contour, i, (0, 255, 0), 3)
            # cv2.putText(vidClone, str(area), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
            arr=np.array([area])
            arr=np.reshape(arr,(-1,1))
            time = model.predict(arr)
            #print(time)
            global TIME
            TIME = int(time)
    cv2.imshow("vidClone", vidClone)
    return vidClone
    keypress = cv2.waitKey(1) & 0xFF
    # if the user pressed "q", then stop looping
    if keypress == ord('q'):
        return


if __name__ == "__main__":
    vid1 = cv2.VideoCapture('latestData.mp4')
    vid2 = cv2.VideoCapture('latestData.mp4')
    vid3 = cv2.VideoCapture('latestData.mp4')
    vid4 = cv2.VideoCapture('latestData.mp4')
    # global refIm
    temp = refIm.copy()
    temp1 = temp.copy()
    for i in range(232):
        for j in range(302):
            temp[i][j] = 0
            temp1[i][j] = 255
    timer = temp.copy()
    # setting the video frame#
    lane1_start_time = calcFrame(1, 60)
    lane1_end_time = calcFrame(2, 36)
    vid1.set(1, lane1_start_time)
    _, frame1 = vid1.read()
    lane2_start_time = calcFrame(2, 52)
    lane2_end_time = calcFrame(3, 22)
    vid2.set(1, lane2_start_time)
    _, frame2 = vid1.read()
    lane3_start_time = calcFrame(6, 56)
    lane3_end_time = calcFrame(7, 26)
    vid3.set(1, lane3_start_time)
    _, frame3 = vid1.read()
    lane4_start_time = calcFrame(12, 22)
    lane4_end_time = calcFrame(12, 52)
    vid4.set(1, lane4_start_time)
    _, frame4 = vid1.read()
    # display window
    st0 = np.hstack((temp, frame1, temp))
    st1 = np.hstack((frame4, timer, frame2))
    st2 = np.hstack((temp, frame3, temp))
    fWin = np.vstack((st0, st1, st2))
    # lane1
    vid1.set(1, calcFrame(2, 15))
    # reading the reference image#
    while vid1.get(1) <= (lane1_end_time):

        ret1, frame1 = vid1.read()
        frame1 = process(frame1)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        fWin = np.vstack((st0, st1, st2))
        # if vid1.get(1)==calcFrame(2,20):
        # t=thread()
        if vid1.get(1)==calcFrame(2,22):
            _t=SetTimer()
            _t.start()
            _t.join()
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane2

    while vid2.get(1) <= (lane2_end_time):

        ret1, frame2 = vid2.read()
        frame2 = process(frame2)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        if vid2.get(1)==calcFrame(3,19):
            _t=SetTimer()
            _t.start()
            _t.join()
        fWin = np.vstack((st0, st1, st2))
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane3

    while vid3.get(1) <= (lane3_end_time) and TIME:

        ret1, frame3 = vid3.read()
        frame3 = process(frame3)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        # for i in range(TIME):
        # cv2.putText(timer,str(TIME-i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
        # time.sleep(1)
        # timer=temp.copy()
        # cv2.imshow('st1',st1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        st2 = np.hstack((temp, frame3, temp))
        if vid3.get(1)==calcFrame(7,22):
            _t=SetTimer()
            _t.start()
            _t.join()
        fWin = np.vstack((st0, st1, st2))
        x, y = 400, 400
        # for i in range(TIME):
        #   print(TIME-i)
        cv2.putText(fWin, str(TIME), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        # time.sleep(1)

        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane4

    while vid4.get(1) <= (lane4_end_time):

        ret1, frame4 = vid4.read()
        frame4 = process(frame4)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        if vid4.get(1)==calcFrame(12,35):
            _t=SetTimer()
            _t.start()
            _t.join()
        fWin = np.vstack((st0, st1, st2))
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

cv2.destroyAllWindows()
