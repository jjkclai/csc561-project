# In[]:

import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filePath = "input your input file path here"
fileName = "input your input file name here"
fileName_ = fileName[:fileName.rfind('.')] + "_dense" + fileName[fileName.rfind("."):]

if filePath[-1] != "/":
    filePath = filePath + "/"

#cv.setNumThreads(8)

# In[]:

def interpolate(img, flow):
    flow[:, :, 0] /= 2.0
    flow[:, :, 1] /= 2.0
    flow[:, :, 0] += np.arange(cap_width)
    flow[:, :, 1] += np.arange(cap_height)[:, np.newaxis]
    result = cv.remap(img, flow, None, cv.INTER_CUBIC, borderMode = cv.BORDER_REPLICATE)
    
    return result

# In[]:

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture(filePath + fileName)

# cap_fps = fps of video, cap_height = height of video, cap_width = width of video
cap_fps = cap.get(cv.CAP_PROP_FPS)
cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

# In[]:

# Initialize video writer
wri = cv.VideoWriter(filePath + fileName_, cv.VideoWriter_fourcc(*"mp4v"), cap_fps * 2, (int(cap_width), int(cap_height)), 1)

# If video writter is unsuccessful
if not wri:
    
    # The following frees up resources and closes all windows
    cv.destroyAllWindows()
    cap.release()
    wri.release()
    
    # stop script
    exit()

# In[]:

# Retrieve start time
time_start = time.time()

# ret = a boolean return value from getting the frame, frame = the first frame in the entire video sequence
ret, frame_prev = cap.read()

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)

# Write frame to video writer
wri.write(frame_prev)

# Continue until all frames in video has been read
while(1):
    
    # ret = a boolean return value from getting the frame, frame = the next frame in the video sequence
    ret, frame = cap.read()
    
    # Stop when frame read unsuccessful
    if not ret:
        break
    
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow
    flow = cv.calcOpticalFlowFarneback(gray_prev, gray, None, 0.5, 3, 31, 3, 7, 1.5, 0)
    
    # Interpolate frame
    frame_interpolated_1 = interpolate(frame_prev, -1 * flow)
    frame_interpolated_2 = interpolate(frame, flow)
    
    # Blend the two interpolated frame
    frame_interpolated = cv.addWeighted(frame_interpolated_1, 0.5, frame_interpolated_2, 0.5, 0.0)
    
    # Write frame to video writer
    wri.write(frame_interpolated)
    wri.write(frame)
    
    # Update previous frame
    frame_prev = frame
    
    # Update previous gray-scale frame
    gray_prev = gray
    
    # Run single operation, for debugging only
    #break
    
# Retrieve end time
time_end = time.time()

# Calculate execution time
print("total execution time: ", time_end - time_start)

# In[]:

# The following frees up resources and closes all windows
cv.destroyAllWindows()
cap.release()
wri.release()
