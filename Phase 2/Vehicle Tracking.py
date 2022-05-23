#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import label
from IPython.display import HTML
from moviepy.editor import VideoFileClip

get_ipython().run_line_magic('matplotlib', 'inline')

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# In[2]:


# Read in cars and notcars
carpics = glob.glob('vehicles/*.png')
notcarpics = glob.glob('non-vehicles/*.png')
all_cars = []
all_notcars = []
cars = []
notcars = []
for image in carpics:
    all_cars.append(image)
for image in notcarpics:
    all_notcars.append(image)
    
# Get only 1/5 of the training data to avoid overfitting
for ix, car in enumerate(all_cars):
    if ix % 5 == 0:
        cars.append(car)
        
for ix, notcar in enumerate(all_notcars):
    if ix % 5 == 0:
        notcars.append(notcar)
        
print('Vehicle train image count: {}'.format(len(all_cars)))
print('Cars train image count: {}'.format(len(cars)))
print('Non-vehicle train image count: {}'.format(len(all_notcars)))
print('Not-Cars train image count: {}'.format(len(notcars)))


# In[3]:


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
# Check the score of the SVC
# Check the prediction time for a single sample
t=time.time()


# In[4]:


model_pickle = {}
model_pickle['svc'] = svc
model_pickle['scaler'] = X_scaler
model_pickle['orient'] = orient
model_pickle['pix_per_cell'] = pix_per_cell
model_pickle['cell_per_block'] = cell_per_block
model_pickle['spatial_size'] = spatial_size
model_pickle['hist_bins'] = hist_bins
pickle.dump( model_pickle, open( "training.pkl", "wb" ))


# In[5]:


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# In[6]:


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# In[7]:


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# In[8]:


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# In[9]:


def draw_labeled_bboxes(img, labels):
    box_list = []
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
        
        box_list.append(bbox)

    return img, box_list


# In[10]:


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    bboxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    
    #add up heat overlap 
    heatmap = add_heat(heat, bboxes)
                
    return heatmap, bboxes 


# In[11]:


def track_vehicles(img):
    
    global draw_img_prev, bbox_list_prev, labels_prev, heatmap_sum
    global first_frame, frame_count
    
    model_pickle = pickle.load(open('training.pkl', 'rb'))
    svc = model_pickle['svc']
    X_scaler = model_pickle['scaler']
    orient = model_pickle['orient']
    pix_per_cell = model_pickle['pix_per_cell'] 
    cell_per_block = model_pickle['cell_per_block']
    spatial_size = model_pickle['spatial_size']
    hist_bins = model_pickle['hist_bins']
    
    #this could be changed relative to image size
    ystart = 400
    ystop = 420
    
    if first_frame:
        
        #initialize the running average heatmap
        heatmap_sum = np.zeros_like(img[:,:,0]).astype(np.float)
        
        for scale in (1.0,1.5,2.0):
            
            #as the image scale gets bigger, the ROI needs to extend
            ystop = ystop + 75

            heatmap, bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                                        pix_per_cell, cell_per_block, spatial_size, hist_bins)
            #sum up heatmap for all scales
            heatmap_sum = np.add(heatmap_sum,heatmap)
        
        heatmap = apply_threshold(heatmap_sum, 2)
        heatmap = np.clip(heatmap, 0, 1)
        labels = label(heatmap)
        draw_img, bbox_list = draw_labeled_bboxes(np.copy(img), labels)

        draw_img_prev = draw_img
        bbox_list_prev = bbox_list
        labels_prev = labels

        first_frame = False
    
        return draw_img
    
    if frame_count <= 2:
        
        frame_count += 1
        
        #reset ROI
        ystop = 420
        
        for scale in (1.0,1.5,2.0):

            ystop = ystop + 75
            heatmap, bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                                                            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            heatmap_sum = np.add(heatmap_sum,heatmap)
            draw_img = np.copy(img)
        
        #draw old boxes
        for unique_car in range(1, labels_prev[1]+1):
            draw_img = cv2.rectangle(draw_img, bbox_list_prev[unique_car-1][0],
                                     bbox_list_prev[unique_car-1][1], (255,0,0), 6)
        return draw_img
    
    heatmap = apply_threshold(heatmap_sum,2)
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)   
    draw_img, bbox_list = draw_labeled_bboxes(np.copy(img), labels)   
    
    draw_img_prev = draw_img
    bbox_list_prev = bbox_list
    labels_prev = labels
    
    #reset heatmap sum and frame_count
    heatmap_sum = np.zeros_like(img[:,:,0]).astype(np.float)
    frame_count = 0
    
    return draw_img


# In[12]:


global first_frame, frame_count
frame_count = 0
first_frame = True

project_video_output = 'Tracking Project Video output.mp4'
clip2 = VideoFileClip('project_video.mp4').subclip(0,30)

project_video_clip = clip2.fl_image(track_vehicles)
get_ipython().run_line_magic('time', 'project_video_clip.write_videofile(project_video_output, audio=False, verbose=False)')


# In[13]:


video = 'Tracking Project Video output.mp4'
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video))


# In[14]:


global first_frame, frame_count
frame_count = 0
first_frame = True

project_video_output = 'Tracking challenge video output.mp4'
clip2 = VideoFileClip('challenge_video.mp4')

project_video_clip = clip2.fl_image(track_vehicles)
get_ipython().run_line_magic('time', 'project_video_clip.write_videofile(project_video_output, audio=False, verbose=False)')


# In[15]:


video = 'Tracking challenge video output.mp4'
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video))

