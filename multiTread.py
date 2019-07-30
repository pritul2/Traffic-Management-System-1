import matplotlib
from threading import *
from time import sleep
import numpy as np
matplotlib.use("TKAgg")
print(matplotlib.get_backend())

from matplotlib import pyplot as plt
import cv2

def fun(x):
    pass
#It convert the time into frame number#
def calcFrame(x,y):
    frame_time=int((x*60+y)*30)
    return frame_time

refIm = cv2.imread('refFrame.jpg')
temp=refIm.copy()
temp1=temp.copy()
for i in range(232):
        for j in range(302):
            temp[i][j]=0
            temp1[i][j]=1
            
#to create window frame
st0=np.hstack((temp,temp1,temp))
st1=np.hstack((temp1,temp,temp1))
st2=np.hstack((temp,temp1,temp))
#cv2.imshow('fWin',fWin)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
(h, w) = st0.shape[:2]
print(h,w)
class lane1(Thread):
    lane1_start_time = calcFrame(1, 60)
    lane1_end_time = calcFrame(2, 35)
    vid1 = cv2.VideoCapture('latestData.mp4')
    vid1.set(1, lane1_start_time)
    def run(self):
        while vid1.get(1) <= (lane1_end_time+1000):
            print("lane1")
            ret1, frame1 = vid1.read()
            global st0
            st0[:232,302:604]=frame1
            cv2.imshow("frame1",st0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

class lane2(Thread):
    lane2_start_time = calcFrame(2, 52)
    lane2_end_time = calcFrame(3, 22)
    vid2 = cv2.VideoCapture('latestData.mp4')
    vid2.set(1, lane2_start_time)

    def run(self):
        while vid2.get(1) <= (lane2_end_time + 1000):
            print("lane2")
            ret2, frame2 = vid2.read()
            st1[:232,0:302]=frame2
            #cv2.imshow("frame2", frame2)
            #cv2.waitKey(1)


class lane3(Thread):
    lane3_start_time = calcFrame(6, 56)
    lane3_end_time = calcFrame(7, 26)
    vid3 = cv2.VideoCapture('latestData.mp4')
    vid3.set(1, lane3_start_time)

    def run(self):
        while vid3.get(1) <= (lane3_end_time + 1000):
            print("lane3")
            ret3, frame3 = vid3.read()
            st1[:232,605:906]=frame3
            #cv2.imshow("frame3", frame3)
            #cv2.waitKey(1)

class lane4(Thread):
    lane4_start_time = calcFrame(12, 22)
    lane4_end_time = calcFrame(12, 52)
    vid4 = cv2.VideoCapture('latestData.mp4')
    vid4.set(1, lane4_start_time)

    def run(self):
        while vid4.get(1) <= (lane4_end_time + 1000):
            print("lane4")
            ret4, frame4 = vid4.read()
            st2[:232,303:604]=frame4
            #cv2.imshow("frame4", frame4)
            #cv2.waitKey(1)

cv2.namedWindow("window")
cv2.createTrackbar("kernel", "window", 3, 1000, fun)
cv2.createTrackbar("threshold", "window", 53, 1000, fun)
cv2.createTrackbar("dilate", "window", 15, 1000, fun)

if __name__ == "__main__":

    refIm = cv2.imread('refFrame.jpg')
    refIm2 = cv2.cvtColor(refIm, cv2.COLOR_BGR2GRAY)

    #Time for the lane1#
    lane1_start_time=calcFrame(1,55)
    lane1_end_time=calcFrame(2,30)
    vid1 = cv2.VideoCapture('latestData.mp4')
    vid1.set(1,lane1_start_time)

    # Time for the lane2#
    lane2_start_time = calcFrame(2,52)
    lane2_end_time = calcFrame(3,22)
    vid2= cv2.VideoCapture('latestData.mp4')
    vid2.set(1,lane2_start_time)

    # Time for the lane3#
    lane3_start_time = calcFrame(6,56)
    lane3_end_time = calcFrame(7,26)
    vid3= cv2.VideoCapture('latestData.mp4')
    vid3.set(1,lane3_start_time)

    # Time for the lane4#
    lane4_start_time = calcFrame(12,22)
    lane4_end_time = calcFrame(12,52)
    vid4= cv2.VideoCapture('latestData.mp4')
    vid4.set(1,lane4_start_time)

    l1=lane1()
    l2 = lane2()
    l3 = lane3()
    l4=lane4()

    l1.start()
    l2.start()
    l3.start()
    l4.start()
    #cv2.waitKey(1)
    #plt.show()

    fWin=np.vstack((st0,st1,st2))
    cv2.imshow('fWin',fWin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    while (vid.isOpened()):
        ret, frame = vid.read()
        (h, w) = frame.shape[:2]
        # print(h,w)
        vidClone = frame.copy()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            y = type(gray)
            # print(y)
            bg = refIm2.copy().astype('float')
            diff = cv2.absdiff(bg.astype('uint8'), gray)

            thresh = cv2.getTrackbarPos("threshold", "window")
            thresholded = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]

            cv2.imshow('Input', gray)
            cv2.imshow('Original', frame)
            cv2.imshow('Dif', thresholded)

            k = 3
            if k % 2 == 0:
                k = k + 1
            kernel = np.ones((k, k), "uint8")
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            cv2.imshow('opening', opening)

            dilate = cv2.getTrackbarPos("dilate", "window")
            dilated = cv2.dilate(opening, None, iterations=dilate)

            cv2.imshow("dilated", dilated)
            contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contour)):
                z = cv2.drawContours(vidClone, contour, i, (0, 255, 0))
                M = cv2.moments(contour[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area = cv2.contourArea(contour[i])
                cv2.putText(vidClone, str(area), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("video clone", vidClone)
            keypress = cv2.waitKey(30) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord('q'):
                break
# workbook.close()'''
#vid.release
#cv2.destroyAllWindows()