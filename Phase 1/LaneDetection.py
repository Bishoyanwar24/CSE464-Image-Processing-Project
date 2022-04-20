#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os
import sys
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks_cwt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

pi = 3.14159
data = pickle.load( open( "camera_calibration.pkl", "rb" ) )
mtx_camera = data[0]
dist_camera = data[1]

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


nx=9
ny=6
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.




# Make a list of calibration images
#plt.figure(figsize = (10,5))
mtx_all = []
dist_all = []
for i in range(20):
    fname = 'camera_cal/calibration'+ str(i+1) + '.jpg'
    img = cv2.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    #plt.subplot(4,5,i+1)
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, corners, gray.shape[::-1],None,None)
        objpoints.append(objp)
        imgpoints.append(corners)
        
        plt.imshow(img)

    else:
        plt.imshow(img)
    plt.axis('off')


# In[3]:


cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

objpoints.append(objp)
imgpoints.append(corners)
plt.imshow(img)


# In[4]:


fname = 'camera_cal/calibration'+ str(1) + '.jpg'
img = cv2.imread(fname)
img_size = (img.shape[1], img.shape[0])


# In[5]:


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


# In[6]:


data = [mtx,dist]
pickle.dump( data, open( "camera_calibration.pkl", "wb" ) )


# In[7]:


corners[1][0][0]


# In[8]:


for i in range(20):
    fname = 'camera_cal/calibration'+ str(i+1) + '.jpg'
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    plt.figure(figsize=(10,4))


    undist = cv2.undistort(img, mtx, dist, None, mtx)
    plt.subplot(1,2,1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    ret_ud, corners_ud = cv2.findChessboardCorners(undist, (nx, ny), None)
 
    if ret == True:
        for i_c in range(len(corners)):
            plt.plot(corners[i_c][0][0],corners[i_c][0][1],'ro')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(undist)
    if ret_ud == True:
        for i_c in range(len(corners_ud)):
            plt.plot(corners_ud[i_c][0][0],corners_ud[i_c][0][1],'bs')
    plt.axis('off');
    plt.show();


# ## Simple Perception Stack for Self-Driving Cars
# 
# Self-driving cars have piqued human interest for centuries. Leonardo Da Vinci sketched out the plans for a hypothetical self-driving cart in the late 1400s, and mechanical autopilots for airplanes emerged in the 1930s. In the 1960s an autonomous vehicle was developed as a possible moon rover for the Apollo astronauts. A true self-driving car has remained elusive until recently. Technological advancements in global positioning (GPS), digital mapping, computing power, and sensor systems have finally made it a reality
# In this project we are going to create a simple perception stack for self-driving cars (SDCs.) Although a typical perception stack for a self-driving car may contain different data sources from different sensors (ex.: cameras, lidar, radar, etc…), we’re only going to be focusing on video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks. The project is split into two phases. We’ll be going into each of them in the following parts.
# ### Phase 1 - Lane Line detection:
# In this first phase, your goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. You’re required to find and track the lane lines and the position of the car from the center of the lane. As a bonus, you can track the radius of curvature of the road too. By all means, you are welcome and encouraged to use any technique that you see fit. You can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane
# 
# <img src='images/car_init_final.png'>
# 
# In the notebook below, we will go over the steps taken to go from the image in the left panel to the image in the right panel. 
# 
# ### The algorithm
# 
# The algorithm is divided into two steps, in the first step we apply a perspective transform and compute a lane mask to identify potential locations of lane in an image, and in the next step we combine the lane mask information with previous frame information to compute the final lane. The second step is performed to discard effects of noisy or 
# 
# #### Part 1: Get lane mask
# 
# Figure below presents the steps involved in obtaining lane masks from the original image. The steps are divided as follows,
# 
# 1. Read and undistort image: In this step, a new image is read by the program and the image is undistorted using precomputed camera distortion matrices. 
# 2. Perspective transform: Read in new image and apply perspective transform. Perspective transformation gives us bird's eye view of the road, this makes further processing easier as any irrelevant information about background is removed from the warped image. 
# 3. Color Masks: Once we obtain the perspective transform, we next apply color masks to identify yellow and white pixels in the image. Color masks are applied after converting the image from RGB to HSV space. HSV space is more suited for identifying colors as it segements the colors into the color them selves (Hue), the ammount of color (Saturation) and brightness (Value). We identify yellow color as the pixels whose HSV-transformed intensities are between \\([ 0, 100, 100]\\) and \\([ 50, 255, 255]\\), and white color as the pixels with intensities between \\( [20,   0,   180]\\) and \\([255,  80, 255] \\).
# 4. Sobel Filters: In addition to the color masks, we apply sobel filters to detect edges. We apply sobel filters on L and S channels of image, as these were found to be robust to color and lighting variations. After multiple trial and error, we decided to use the magnitude of gradient along x- and y- directions with thresholds of 50 to 200 as good candidates to identify the edges. 
# 5. Combine sobel and color masks: In a final step we combined candidate lane pixels from Sobel filters  and color masks to obtain potential lane regions. 
# 
# These  steps are illustrated in the figure below. 
# 
# <img src='images/lane_mask.png'>
# 
# 
# From above, we get good representation of lane masks. However, these masks are based on yellow and white colors and sobel filter calculations. If there are additional drawings or markings on the road, this algorithm will not give two neat lines as above, we will therefore perform additional analysis to isolate the lane loactions. 
# 
# #### Part 2: Compute lanes 
# 
# We implement different lane calculations for the first frame and subsequent frames. In the first frame, we compute the lanes using computer vision methods, however, in the later frames, we skip these steps. Instead, we place windows of 50 pixel width centered on the lanes computed in the previous frame, and search within these windows. This significanly reduced the computation time, for our algorithm. We were able to achieve 10 Frames/s lane estimation rate. 
# 
# 
# #### Compute lanes for the first frame
# 
# The next step is to compute lanes for the first image. To do so, we take the lane mask from the previous step, and take only the bottom half of the image. We next use scipy to compute the locations of the peaks corresponding to the left and right lanes. 
# 
# 
# <img src='images/hist_lane1.png'>
# 
# We then place a window of size 50 pixels centered at these peaks, and search for peaks in the bottom 8th of the image. Next we move up to the next 1/8th of the image and center windows at the peaks detected in the bottom 1/8th of the image. We repeat this process 8 times to cover the entire image. This is illustrated in the figure below. 
# 
# 
# <img src='images/road_slices.png'>
# 
# 
# In addition to tracking the location of the previous window, we also keep track of the displacement of previous window. In cases where no peaks are found, we place a window centered at the location calculated assuming the location of previous window moved by a precomputed offset. The windows and lanes obtained after this step are shown below.  
# 
# <img src='images/sliding_window.png'>
# 
# 
# 
# 
# We next fit a quadratic function with independent variable 'x' and dependent variable 'y' to the points within the line mask using numpy's polyfit function. 
# 
# <img src='images/poly_fit.png'>
# 
# After computing the lanes, we draw them back on the original undistorted image as follows. 
# 
# <img src='images/lane_draw.png'>
# 
# 
# 
# #### Part 3: Check lane mask against previous frame, and compute new lane regions. 
# 
# 
# If the current frame is not the first frame, we follow the same steps as part 2 to get the lane masks, however, we introduced additional steps to ensure any error due to incorrectly detected lanes is removed. Lane correction are introduced as, 
# 
# 1. Outlier removal 1: If the change in coefficient is above 0.005, the lanes are discarded. This number was obtained empirically. 
# 2. Outlier removal 2: If any lane was found with less than 5 pixels, we use the previous line fit coefficients as the coefficients for the current one. 
# 3. Smoothing: We smooth the value of the current lane using a first order filter response, as \\(coeffs = 0.95*coeff~prev+ 0.05 coeff\\). 
# 
# Finally , we use the coefficients of polynomial fit to compute curvatures of the lane, and relative location of the car in the lane. 
# 
# 
# ### Reflection: 
# 
# This was a very interesting and fun project to do. The most interesting part was to see how the techniques developed in a previous simpler project applied to a more real-life type scenario. The work on this project is far from over. The current algorithm is not robust enough to generalize to challenge videos, but performs remarkably well. We will go over details of a more robust algorithm in our final report.
# 

# In[ ]:





# In[9]:


# Define kernel size for filters
kernel_size = 5
# Define size for sliding windows
window_size = 60


# debug; variable to 


# In[10]:


#Functions

def draw_pw_lines(img,pts,color):
    # This function draws lines connecting 10 points along the polynomial
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i+1][0]
        y2 = pts[0][i+1][1]
        cv2.line(img, (x1, y1), (x2, y2),color,50)
        
def undistort_image(img, mtx, dist):
    # Function to undistort image
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img

def gaussian_blur(img, kernel=5):
    # Function to smooth image
    blur = cv2.GaussianBlur(img,(kernel,kernel),0)
    return blur

def get_curvature(pol_a,y_pt):
    # Returns curvature of a quadratic
    A = pol_a[0]
    B = pol_a[1]
    R_curve = (1+(2*A*y_pt+B)**2)**1.5/2/A
    return R_curve

def stack_arr(arr):
    # Stacks 1-channel array into 3-channel array to allow plotting
    return np.stack((arr, arr,arr), axis=2)

def apply_perspective_transform(image):
    # Applies bird-eye perspective transform to an image
    img_size = image.shape
    ht_window = np.uint(img_size[0]/1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1]/2)
    ctl_window = c_window - .25*np.uint(img_size[1]/2)
    ctr_window = c_window + .25*np.uint(img_size[1]/2)
    cbl_window = c_window - 1*np.uint(img_size[1]/2)
    cbr_window = c_window + 1*np.uint(img_size[1]/2)
    src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],
                      [ctr_window,ht_window],[ctl_window,ht_window]])
    dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],
                  [img_size[1],0],[0,0]])
    
    warped,M_warp,Minv_warp = warp_image(image,src,dst,(img_size[1],img_size[0])) # returns birds eye image
    return warped,M_warp,Minv_warp

def get_initial_mask(img,window_sz):
    
    # This function gets the initial mask
    
    img = gaussian_blur(img,5)
    img_size = np.shape(img)
    mov_filtsize = int(img_size[1]/50.)
    mean_ln = np.mean(img[int(img_size[0]/2):,:],axis=0)
    mean_ln = moving_average(mean_ln,mov_filtsize)
    
    indexes = find_peaks_cwt(mean_ln,[100], max_distances=[800])

    val_ind = np.array([mean_ln[indexes[i]] for i in range(len(indexes)) ])
    ind_sorted = np.argsort(-val_ind)

    ind_peakR = indexes[ind_sorted[0]]
    ind_peakL = indexes[ind_sorted[1]]
    if ind_peakR<ind_peakL:
        ind_temp = ind_peakR
        ind_peakR = ind_peakL
        ind_peakL = ind_temp

    n_vals = 8
    ind_min_L = ind_peakL-window_sz
    ind_max_L = ind_peakL+window_sz

    ind_min_R = ind_peakR-window_sz
    ind_max_R = ind_peakR+window_sz

    mask_L_i = np.zeros_like(img)
    mask_R_i = np.zeros_like(img)

    ind_peakR_prev = ind_peakR
    ind_peakL_prev = ind_peakL
    
    # Split image into 8 parts and compute histogram on each part
    
    for i in range(8):
        img_y1 = int(img_size[0]-img_size[0]*i/8)
        img_y2 = int(img_size[0]-img_size[0]*(i+1)/8)
    
        mean_lane_y = np.mean(img[img_y2:img_y1,:],axis=0)
        mean_lane_y = moving_average(mean_lane_y,mov_filtsize)
        indexes = find_peaks_cwt(mean_lane_y,[100], max_distances=[800])
        
        if len(indexes)>1.5:
            val_ind = np.array([mean_ln[indexes[i]] for i in range(len(indexes)) ])
            ind_sorted = np.argsort(-val_ind)

            ind_peakR = indexes[ind_sorted[0]]
            ind_peakL = indexes[ind_sorted[1]]
            if ind_peakR<ind_peakL:
                ind_temp = ind_peakR
                ind_peakR = ind_peakL
                ind_peakL = ind_temp
            
        else:
        # If no pixels are found, use previous ones. 
            if len(indexes)==1:
                if (np.abs(indexes[0]-ind_peakR_prev)<np.abs(indexes[0]-ind_peakL_prev)):
                    ind_peakR = indexes[0]
                    ind_peakL = ind_peakL_prev
                else:
                    ind_peakL = indexes[0]
                    ind_peakR = ind_peakR_prev
            else:
                ind_peakL = ind_peakL_prev
                ind_peakR = ind_peakR_prev
            
            
        # If new center is more than 60pixels away, use previous
        # Outlier rejection
        if np.abs(ind_peakL-ind_peakL_prev)>=60:
            ind_peakL = ind_peakL_prev

        if np.abs(ind_peakR-ind_peakR_prev)>=60:
            ind_peakR = ind_peakR_prev
            
    
            
        mask_L_i[img_y2:img_y1,ind_peakL-window_sz:ind_peakL+window_sz] = 1.
        mask_R_i[img_y2:img_y1,ind_peakR-window_sz:ind_peakR+window_sz] = 1.
        
        ind_peakL_prev = ind_peakL
        ind_peakR_prev = ind_peakR
        
    return mask_L_i,mask_R_i

def get_mask_poly(img,poly_fit,window_sz):
    
    # This function returns masks for points used in computing polynomial fit. 
    mask_poly = np.zeros_like(img)
    img_size = np.shape(img)

    poly_pts = []
    pt_y_all = []

    for i in range(8):
        img_y1 = img_size[0]-img_size[0]*i//8
        img_y2 = img_size[0]-img_size[0]*(i+1)//8

        pt_y = (img_y1+img_y2)/2
        pt_y_all.append(pt_y)
        poly_pt = np.round(poly_fit[0]*pt_y**2 + poly_fit[1]*pt_y + poly_fit[2])
    
        poly_pts.append(poly_pt)
        fx_1 = int(poly_pt-window_sz)
        fx_2 = int(poly_pt+window_sz)
    
        mask_poly[img_y2:img_y1,fx_1:fx_2] = 1.     

    return mask_poly, np.array(poly_pts),np.array(pt_y_all)

def get_val(y,pol_a):
    # Returns value of a quadratic polynomial 
    return pol_a[0]*y**2+pol_a[1]*y+pol_a[2]

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient=='x':
        img_s = cv2.Sobel(img,cv2.CV_64F, 1, 0)
    else:
        img_s = cv2.Sobel(img,cv2.CV_64F, 0, 1)
    img_abs = np.absolute(img_s)
    img_sobel = np.uint8(255*img_abs/np.max(img_abs))
    
    binary_output = 0*img_sobel
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    img_sx = cv2.Sobel(img,cv2.CV_64F, 1, 0)
    img_sy = cv2.Sobel(img,cv2.CV_64F, 0, 1)
    
    img_s = np.sqrt(img_sx**2 + img_sy**2)
    img_s = np.uint8(img_s*255/np.max(img_s))
    binary_output = 0*img_s
    binary_output[(img_s>=thresh[0]) & (img_s<=thresh[1]) ]=1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    img_sx = cv2.Sobel(img,cv2.CV_64F,1,0, ksize=sobel_kernel)
    img_sy = cv2.Sobel(img,cv2.CV_64F,0,1, ksize=sobel_kernel)
    
    grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))
    
    binary_output = 0*grad_s # Remove this line
    binary_output[(grad_s>=thresh[0]) & (grad_s<=thresh[1])] = 1
    return binary_output

def GaussianC_Adaptive_Threshold(img,kernel,cut_val):
    # Gaussian adaptive thresholding (NOT USED )
    img_cut = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,kernel,cut_val)
    return img_cut

def warp_image(img,src,dst,img_size):
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped,M,Minv

def sobel_combined(image):
    # Combine sobel masks.
    img_g_mag = mag_thresh(image,3,(20,150))
    img_d_mag = dir_threshold(image,3,(.6,1.1))
    img_abs_x = abs_sobel_thresh(image,'x',5,(50,200))
    img_abs_y = abs_sobel_thresh(image,'y',5,(50,200))
    sobel_combined = np.zeros_like(img_d_mag)
    sobel_combined[((img_abs_x == 1) & (img_abs_y == 1)) | \
               ((img_g_mag == 1) & (img_d_mag == 1))] = 1
    return sobel_combined

def color_mask(hsv,low,high):
    # Takes in low and high values and returns mask
    mask = cv2.inRange(hsv, low, high)
    return mask

def apply_color_mask(hsv,img,low,high):
    # Takes in color mask and returns image with mask applied.
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

def moving_average(a, n=3):
    # Moving average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[11]:


def pipeline_process_highway(image):
    
    global left_fit_prev   
    global right_fit_prev
    global col_R_prev
    global col_L_prev
    global set_prev
    global mask_poly_L
    global mask_poly_R
    

    # Undistort image

    image = undistort_image(image, mtx_camera , dist_camera )
    image = gaussian_blur(image, kernel=5)
    img_size = np.shape(image)
    
    # Define window for perspective transform
    warped,M_warp,Minv_warp = apply_perspective_transform(image)
    image_HSV = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)

    # Define color ranges and apply color mask
    yellow_hsv_low  = np.array([ 0, 100, 100])
    yellow_hsv_high = np.array([ 50, 255, 255])

    white_hsv_low  = np.array([  20,   0,   180])
    white_hsv_high = np.array([ 255,  80, 255])
    # get yellow and white masks 
    mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
    # Combine white and yellow masks into 1
    mask_lane = cv2.bitwise_or(mask_yellow,mask_white) 
    
    # Convert image to HLS scheme
    image_HLS = cv2.cvtColor(warped,cv2.COLOR_RGB2HLS)

    # Apply sobel filters on L and S channels.
    img_gs = image_HLS[:,:,1]
    img_abs_x = abs_sobel_thresh(img_gs,'x',5,(50,225))
    img_abs_y = abs_sobel_thresh(img_gs,'y',5,(50,225))
    wraped2 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))
    
    img_gs = image_HLS[:,:,2]
    img_abs_x = abs_sobel_thresh(img_gs,'x',5,(50,255))
    img_abs_y = abs_sobel_thresh(img_gs,'y',5,(50,255))
    wraped3 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))
    

    # Combine sobel filter information from L and S channels.
    image_cmb = cv2.bitwise_or(wraped2,wraped3)
    image_cmb = gaussian_blur(image_cmb,25)
    

    # Combine masks from sobel and color masks.

    image_cmb1 = np.zeros_like(image_cmb)
    image_cmb1[(mask_lane>=.5)|(image_cmb>=.5)]=1
    
    
    # If this is first frame, get new mask.
    if set_prev == 0:
        image_cmb1 = gaussian_blur(image_cmb1,5)
        mask_poly_L,mask_poly_R = get_initial_mask(image_cmb1,40)

        
        
    # Define all colors as white to start.         
    col_R = (255,255,255)
    col_L = (255,255,255)
    col_R = (255,255,255)
    col_L = (255,255,255)
    
    # Apply mask to sobel images and compute polynomial fit for left. 
    img_L = np.copy(image_cmb1)
    img_L = cv2.bitwise_and(image_cmb1,image_cmb1,
                                mask = mask_poly_L)
    vals = np.argwhere(img_L>.5)
    if len(vals)<5: ## If less than 5 points 
        left_fit = left_fit_prev
        col_L = col_L_prev
    else:
        all_x = vals.T[0]
        all_y =vals.T[1]
        left_fit = np.polyfit(all_x, all_y, 2)
        if np.sum(cv2.bitwise_and(img_L,mask_yellow))>1000:
            col_L = (255,255,0)
            
    # Apply mask to sobel images and compute polynomial fit for right. 

    img_R = np.copy(image_cmb1)
    img_R = cv2.bitwise_and(image_cmb1,image_cmb1,
                                mask = mask_poly_R)
    vals = np.argwhere(img_R>.5)
        
    if len(vals)<5:
        right_fit = right_fit_prev
        col_R = col_R_prev
    else:
        all_x = vals.T[0]
        all_y =vals.T[1]
        right_fit = np.polyfit(all_x, all_y, 2)
        if np.sum(cv2.bitwise_and(img_R,mask_yellow))>1000:
            col_R = (255,255,0)
    
    
    ## assign initial mask, and save coefficient values for next frame
            
    if set_prev == 0:
        set_prev = 1
        right_fit_prev = right_fit
        left_fit_prev  = left_fit
    
       
    ## Check error between current coefficient and on from previous frame
    err_p_R = np.sum((right_fit[0]-right_fit_prev[0])**2) #/np.sum(right_fit_prev[0]**2)
    err_p_R = np.sqrt(err_p_R)
    if err_p_R>.0005:
        right_fit = right_fit_prev
        col_R = col_R_prev
    else:
        right_fit = .05*right_fit+.95*right_fit_prev
        
    ## Check error between current coefficient and on from previous frame
    err_p_L = np.sum((left_fit[0]-left_fit_prev[0])**2) #/np.sum(right_fit_prev[0]**2)
    err_p_L = np.sqrt(err_p_L)
    if err_p_L>.0005:
        left_fit =  left_fit_prev
        col_L = col_L_prev
    else:
        left_fit =  .05* left_fit+.95* left_fit_prev
    

    ## Compute lane mask for future frame 
    mask_poly_L,left_pts,img_pts = get_mask_poly(image_cmb1,left_fit,window_size)
    mask_poly_R,right_pts,img_pts = get_mask_poly(image_cmb1,right_fit,window_size)
     
        
    ## Compute lanes
        
    right_y = np.arange(11)*img_size[0]/10
    right_fitx = right_fit[0]*right_y**2 + right_fit[1]*right_y + right_fit[2]

    left_y = np.arange(11)*img_size[0]/10
    left_fitx = left_fit[0]*left_y**2 + left_fit[1]*left_y + left_fit[2]
    
    warp_zero = np.zeros_like(image_cmb1).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y])))])
    pts = np.hstack((pts_left, pts_right))
    

    ## Compute intercepts
    left_bot = get_val(img_size[0],left_fit)
    right_bot = get_val(img_size[0],right_fit)
    
    ## Compute center location
    val_center = (left_bot+right_bot)/2.0
    
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    ## Compute lane offset   
    position = (right_bot+left_bot)/2
    distance_from_center = abs((640 - position)*3.7/700) 
    
    # Draw the lane onto the warped blank image    
    draw_pw_lines(color_warp,np.int_(pts_left),col_L)
    draw_pw_lines(color_warp,np.int_(pts_right),col_R)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    
    newwarp = cv2.warpPerspective(color_warp, Minv_warp, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)
    
    # Compute radius of curvature for each lane in meters
    left_curve = get_curvature(left_fit,img_size[0]/2)
    Right_curve = get_curvature(right_fit,img_size[0]/2)
    Total_curvature = abs((left_curve+Right_curve)/2)
    str_curv = 'Raduis of Curvature = ' + str(np.round(Total_curvature,2)) + 'm'
    
    font = cv2.FONT_HERSHEY_COMPLEX    
    
    # Calculate the vehicle position relative to the center of the lane
    if abs(left_curve) > abs(Right_curve):
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (30, 90),
                 fontFace = 16, fontScale = 1, color=(255,255,255), thickness = 2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (30, 90),
                 fontFace = 16, fontScale = 1, color=(255,255,255), thickness = 2)
        
    cv2.putText(result, str_curv, (30,140), fontFace = 16, fontScale = 1, color=(255,255,255), thickness = 2)
    
    right_fit_prev = right_fit
    left_fit_prev  = left_fit
    col_R_prev = col_R
    col_L_prev = col_L
    
    
    #return result    # using cv2 for drawing text in diagnostic pipeline.
    
    if debug == 1:
        font = cv2.FONT_HERSHEY_COMPLEX

    
        # assemble the screen example
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:780, 0:1280] = cv2.resize(result, (1280,780), interpolation=cv2.INTER_AREA)
        #1
        diagScreen[0:240, 1280:1600] = cv2.resize(warped, (320,240), interpolation=cv2.INTER_AREA) 
        #2
        diagScreen[0:240, 1600:1920] = cv2.resize(stack_arr(mask_lane), (320,240), interpolation=cv2.INTER_AREA)
        #3
        diagScreen[240:480, 1280:1600] = cv2.resize(apply_color_mask(image_HSV,warped,yellow_hsv_low,yellow_hsv_high), (320,240), interpolation=cv2.INTER_AREA)
        #4
        diagScreen[240:480, 1600:1920] = cv2.resize(apply_color_mask(image_HSV,warped,white_hsv_low,white_hsv_high), (320,240), interpolation=cv2.INTER_AREA)*4
        #5-big
        diagScreen[600:1080, 1280:1920] = cv2.resize(color_warp, (640,480), interpolation=cv2.INTER_AREA)*4
        #6
        diagScreen[780:1080, 0:320] = cv2.resize(newwarp, (320,300), interpolation=cv2.INTER_AREA)
        #7
        diagScreen[780:1080, 320:640] = cv2.resize(stack_arr(255*image_cmb1), (320,300), interpolation=cv2.INTER_AREA)
        #8
        diagScreen[780:1080, 640:960] = cv2.resize(stack_arr(255*mask_poly_L+255*mask_poly_R), (320,300), interpolation=cv2.INTER_AREA)
        #9
        diagScreen[780:1080, 960:1280] = cv2.resize(stack_arr(255*cv2.bitwise_and(image_cmb1,image_cmb1,mask=mask_poly_L+mask_poly_R)),(320,300), interpolation=cv2.INTER_AREA)
        return diagScreen
    else:
        return result


# In[ ]:





# In[12]:


set_prev = 0
debug = 0

project_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4");
white_clip = clip1.fl_image(pipeline_process_highway) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(project_output, audio=False);')


# In[13]:


set_prev = 0
debug = 1

project_output_diag = 'project_video_debug.mp4'
clip2 = VideoFileClip("project_video.mp4");
white_clip = clip2.fl_image(pipeline_process_highway) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(project_output_diag, audio=False);')


# In[ ]:





# In[14]:


set_prev = 0
debug = 0

challenge_output = 'challenge_video_output.mp4'
clip3 = VideoFileClip("challenge_video.mp4");
white_clip = clip3.fl_image(pipeline_process_highway) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(challenge_output, audio=False);')


# In[15]:


set_prev = 0
debug = 1

challenge_output_diag = 'challenge_video_debug.mp4'
clip4 = VideoFileClip("challenge_video.mp4");
white_clip = clip4.fl_image(pipeline_process_highway) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(challenge_output_diag, audio=False);')

