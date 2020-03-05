# In[]:

import time

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filePath = "input your input file path here"
fileName = "input your input file name here"
fileName_ = fileName[:fileName.rfind('.')] + "_sparse" + fileName[fileName.rfind("."):]

if filePath[-1] != "/":
    filePath = filePath + "/"

#cv.setNumThreads(8)

# In[]:

def getflow(pts_prev, pts, window_size):
    result = np.zeros((int(cap_height), int(cap_width), 2), dtype="float32")
    window_size_ = int((window_size-1)/2)
    for i in range(pts_prev.shape[0]-1, -1, -1):
        xPos = int(pts_prev[i][0])
        yPos = int(pts_prev[i][1])
        
        xRange_L = xPos - window_size_
        if xRange_L < 0:
            xRange_L = 0
        elif xRange_L >= int(cap_width):
            xRange_L = int(cap_width) - 1
            
        xRange_R = xPos + window_size_ + 1
        if xRange_R < 0:
            xRange_R = 0
        elif xRange_R >= int(cap_width):
            xRange_R = int(cap_width) - 1
            
        yRange_L = yPos - window_size_
        if yRange_L < 0:
            yRange_L = 0
        elif yRange_L >= int(cap_height):
            yRange_L = int(cap_height) - 1
            
        yRange_R = yPos + window_size_ + 1
        if yRange_R < 0:
            yRange_R = 0
        elif yRange_R >= int(cap_height):
            yRange_R = int(cap_height) - 1
        
        result[yRange_L:yRange_R][xRange_L:xRange_R] = pts[i] - pts_prev[i]
#        for x in range(xPos - window_size_, xPos + window_size_ + 1):
#            if x < 0 or x >= cap_width: continue
#            for y in range(yPos - window_size_, yPos + window_size_ + 1):
#                if y < 0 or y >= cap_height: continue
#                result[y][x] = pts[i] - pts_prev[i]
                
    return result

# In[]:

def interpolate(img, flow):
    flow = -1 * flow
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

# Parameters for Shi-Tomasi corner detection
params_st = dict(maxCorners = 10000, qualityLevel = 0.01, minDistance = 0, blockSize = 3)
# Parameters for Lucas-Kanade optical flow
params_lk = dict(winSize = (31, 31), maxLevel = 3, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# ret = a boolean return value from getting the frame, frame = the first frame in the entire video sequence
ret, frame_prev = cap.read()

# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)

# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
st_prev = cv.goodFeaturesToTrack(gray_prev, mask = None, **params_st)

while(st_prev is None):
    wri.write(frame_prev)
    wri.write(frame_prev)
    
    ret, frame_prev = cap.read()
    gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
    st_prev = cv.goodFeaturesToTrack(gray_prev, mask = None, **params_st)

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
    
    # Calculate sparse optical flow
    st, status, error = cv.calcOpticalFlowPyrLK(gray_prev, gray, st_prev, None, **params_lk)
    
    if st is None:
        wri.write(frame)
        wri.write(frame)
       
        frame_prev = frame
        gray_prev = gray
        st_prev = cv.goodFeaturesToTrack(gray_prev, mask = None, **params_st)
        
        while(st_prev is None):
            wri.write(frame_prev)
            wri.write(frame_prev)
    
            ret, frame_prev = cap.read()
            gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
            st_prev = cv.goodFeaturesToTrack(gray_prev, mask = None, **params_st)
    
    else:
    
        # Select good feature points for previous frame
        pts_prev = st_prev[status == 1]
    
        # Select good feature points for current frame
        pts = st[status == 1]
    
        # Calculate actual flow
        flow = getflow(pts_prev, pts, 15)
    
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
    
        # Update previous strong corners
        st_prev = pts.reshape(-1, 1, 2)
    
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
