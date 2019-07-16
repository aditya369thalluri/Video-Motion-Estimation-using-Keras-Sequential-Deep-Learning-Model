import cv2
import os
import sys

path = "E:\\Education\\UNMSEM-1\\Digital Image Processing\\FinalAssignment\\Videos"

count=1
for filename in os.listdir(path):
    vidcap = cv2.VideoCapture(os.path.join(path,filename))
    success,image = vidcap.read()
    cv2.imwrite("frame%d.jpg" % count, image)
    count+=1
