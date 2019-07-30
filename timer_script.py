import cv2
import numpy as np
import time

canvas=np.zeros((232,302),"uint8")
for i in range(60):
  cv2.putText(canvas,str(60-i),(int(canvas.shape[0]/2),int(canvas.shape[1]/2)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255))
  time.sleep(1)
  canvas=np.zeros((232,302),"uint8")
