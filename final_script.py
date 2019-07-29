import cv2
import numpy as np
#from sklearn.externals import joblib
def fun(x):
    pass
def calcFrame(x, y):
    frame_time = int((x * 60 + y) * 30)
    return frame_time
cv2.namedWindow("window")
cv2.createTrackbar("x","window",0,1000,fun)
cv2.createTrackbar("y","window",1000,1000,fun)
cv2.createTrackbar("z","window",0,1000,fun)
cv2.createTrackbar("w","window",1000,1000,fun)

if __name__ == "__main__":
    vid = cv2.VideoCapture('latestData.mp4')
    #model = joblib.load("model (1).cpickle")
    # setting the video frame#
    lane1_start_time = calcFrame(1, 60)
    lane1_end_time = calcFrame(2, 35)
    vid.set(1, lane1_start_time)

    # reading the reference image#
    refIm = cv2.imread('refFrame.jpg')
    refIm2 = cv2.cvtColor(refIm, cv2.COLOR_BGR2GRAY)

    while vid.get(1) <= lane1_end_time + 1000:
        ret, frame = vid.read()

        vidClone = frame.copy()
        #vidClone=vidClone[29:89, 33:75]
        #frame = frame[29:89, 33:75]
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Absolute difference#
            bg = refIm2.copy()
            x=cv2.getTrackbarPos("x","window")
            y = cv2.getTrackbarPos("y", "window")
            z = cv2.getTrackbarPos("z", "window")
            w = cv2.getTrackbarPos("w", "window")
            bg=bg[x:y,z:w]
            gray=gray[x:y,z:w]
            print(gray.shape)
            cv2.imshow("background",bg)
            cv2.waitKey(1)
            diff = cv2.absdiff(bg.astype('uint8'), gray)
            # threshold logic#
            thresh = 53
            thresholded = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]

            # Opening logic#
            k = 3
            kernel = np.ones((k, k), "uint8")
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            cv2.imshow('opening', opening)

            # dilation logic#
            dilate = 15
            dilated = cv2.dilate(opening, None, iterations=dilate)

            # change to _,contour,_ for latest version#
            contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Finding the area of each contour#
            for i in range(len(contour)):
                z = cv2.drawContours(vidClone, contour, i, (0, 255, 0))
                #area = cv2.contourArea(contour[i])
                M = cv2.moments(contour[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area = cv2.contourArea(contour[i])
                cv2.putText(vidClone, str(area), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("video clone", vidClone)

            #time=model.predict(area)
            #print(time)

            keypress = cv2.waitKey(30) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord('q'):
                break
# workbook.close()
vid.release
cv2.destroyAllWindows()
