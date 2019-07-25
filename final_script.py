import cv2
import numpy as np

def fun(x):
    pass
cv2.namedWindow("window")
cv2.createTrackbar("kernel","window",3,1000,fun)
cv2.createTrackbar("threshold","window",53,1000,fun)
cv2.createTrackbar("dilate","window",15,1000,fun)



if __name__ == "__main__":
    top, right, bottom, left = 5, 10, 280, 180
    vid = cv2.VideoCapture('latestData.mp4')
    refIm = cv2.imread('refFrame.jpg')
    refIm2 = cv2.cvtColor(refIm, cv2.COLOR_BGR2GRAY)
  
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

            thresh=cv2.getTrackbarPos("threshold","window")
            thresholded = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
           
            cv2.imshow('Input', gray)
            cv2.imshow('Original', frame)
            cv2.imshow('Dif', thresholded)

            k=cv2.getTrackbarPos("kernel","window")
            if k%2==0:
                k=k+1
            kernel=np.ones((k,k),"uint8")
            opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            cv2.imshow('opening', opening)

            dilate=cv2.getTrackbarPos("dilate","window")
            dilated=cv2.dilate(opening,None,iterations=dilate)

            cv2.imshow("dilated",dilated)
            contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contour)):
                z=cv2.drawContours(vidClone,contour,i,(0,255,0))
                M=cv2.moments(contour[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                area=cv2.contourArea(contour[i])
                cv2.putText(vidClone,str(area),(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

            cv2.imshow("video clone",vidClone)
            keypress = cv2.waitKey(30) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord('q'):
                break
# workbook.close()
vid.release
cv2.destroyAllWindows()
