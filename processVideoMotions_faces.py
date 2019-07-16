# -*- coding: utf-8 -*-
"""
   The code process the video and produces dense video motions plots.
   There is failure in the method and this shown as -inf in the magnitude plot.
Demo taken from:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
"""

import cv2
import numpy as np
import math  as math
import pandas as pd
import pdb
import sys
import os



# draw_flow():
#  im:   the grayscale image to draw on.
#  flow: the (u, v) motion vectors.
#  step: the number of pixels to skip between vectors. 
# Function from: http://programmingcomputervision.com/
def draw_flow(im,flow,step=8):
    """ Plot optical flow at sample points
        spaced step pixels apart. """
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y.astype(int),x.astype(int)].T
        
    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    
    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        mag  = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        ang  = math.degrees(math.atan2(y1-y2, x1-x2))
        
        # Use 1.75 for Eating1Round1Side.mp4
        cond = (mag > 1.5) and (mag < 6) # and ((ang > -45) or (ang < 45))
        if (cond):
            cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    
    return vis

def get_recwh(csv):
    """
    Returns median height and width of rectangle
    """
    med_h = math.ceil(csv['h'].median())
    med_w = math.ceil(csv['w'].median())
    return med_w, med_h








# ------------------------------------ #
# -------- Start Executing ----------- #
# ------------------------------------ #

# Settings
framesToSkip = 5;
reduceFactor = 2;

display_plots = True;

# Parsing command line arguments
if len(sys.argv) != 4:
    print("ERROR: requires two arguments")
    print("USAGE: python3 processVideoMotions.py <video path> <csv path> <class>")
    print("\t video path : path to video file")
    print("\t csv path   : path to csv file having bounding box coordinates")
    print("\t class : Talking or no talking. {'t','nt'}")
    sys.exit(1)

# Printing details
print("Keeping one out of ", framesToSkip, " video frames")
if reduceFactor < 1:
    print("ROI reduced by ", reduceFactor)
else:
    print("ROI increased by", reduceFactor)

# Reading video and ground truth file
cap       = cv2.VideoCapture(sys.argv[1])
csv       = pd.read_csv(sys.argv[2])

# Reading frame 1 / ROI of frame 1
ret, frame1_full = cap.read()
w, h = get_recwh(csv.copy())
csv        = csv.iloc[1:]
csv        = csv.reset_index(drop=True)
frame1_roi = csv.iloc[0]
frame1_roi = frame1_roi
x               = math.floor(frame1_roi['x'])
y               = math.floor(frame1_roi['y'])
frame1 =frame1_full[y:y+h,x:x+w,:] # Cropping face
frame1 = cv2.resize(frame1, (0,0), fx=reduceFactor, fy=reduceFactor)

# Making frame 1 as previous frame
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


firstTime = False
print("Press Esc on the image to exit")
print("Press s   on the image to save")
delay_display = 1  # in milliseconds

mv_img_idx = 0 # index of motion vector image
while(cap.isOpened() and len(csv) > 5):
    for i in range(framesToSkip):
        success_flag, full_frame = cap.read()
        full_frame_roi  = csv.iloc[0]
        full_frame_roi  = full_frame_roi
        x               = math.floor(full_frame_roi['x'])
        y               = math.floor(full_frame_roi['y'])
        full_frame      = full_frame[y:y+h,x:x+w,:] # Cropping face
        csv             = csv.iloc[1:]
        csv             = csv.reset_index(drop=True)
        if (not success_flag):
            break
    
    if (not success_flag):
        break

    print(len(csv))
    color_frame         = cv2.resize(full_frame, (0,0), fx=reduceFactor, fy=reduceFactor) 
    current_frame       = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    flow                = cv2.calcOpticalFlowFarneback(prvs,
                                                       current_frame, None,
                                                       0.5, 3, 15, 3, 5, 1.2, 0)
    # Resizing optical flow to 32x32 matrix
    flow_32             = cv2.resize(flow, (32,32))
    prvs                = current_frame

    # Saving flows
    csv_name = os.path.basename(sys.argv[2])
    csv_name = os.path.splitext(csv_name)[0]
    np_name  = csv_name + "_" + str(mv_img_idx) + ".npy"
    if sys.argv[3] == 't':
        print("Talking")
        np.save("faces/t/"+np_name,flow_32)
    elif sys.argv[3] == 'nt':
        print("Not Talking")
        np.save("faces/nt/"+np_name,flow_32)
     elif sys.argv[3] == 'vt':
        print(" Talking")
        np.save("faces_validation/t/"+np_name,flow_32)
     elif sys.argv[3] == 'vnt':
        print("Not Talking")
        np.save("faces_validation/nt/"+np_name,flow_32)
     elif sys.argv[3] == 'tt':
        print(" Talking")
        np.save("faces_testing/t/"+np_name,flow_32)
     elif sys.argv[3] == 'tnt':
        print("Not Talking")
        np.save("faces_testing/nt/"+np_name,flow_32)
    else:
        print("ERROR: Class type does not exist")
        sys.exit(1)


    # Display motion vectors
    if display_plots:
        mag, ang            = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0]          = ang*180/np.pi/2
        hsv[...,2]          = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr                 = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        motion_vectors = draw_flow(current_frame, flow)
        cv2.imshow('Motion vector plot', motion_vectors)
    
        # Process key pressed:
        #  Esc will escape, s will save the image.
        k = cv2.waitKey(delay_display) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)    
        if (firstTime):
            res = input("Please center the display and hit any key")
            firstTime = False
    mv_img_idx = mv_img_idx + 1

# Release the video
cap.release()

# Close all the OpenCV Windows
cv2.destroyAllWindows()
