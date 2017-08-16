#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
#calibration coefficient
calibrate = tuple()
#last frame's fit lines
first_fit = tuple()
last_fit = tuple()
last_fitx = tuple()
no_lane = False
#boolean for first frame
first_frame = True
#y axis pixel for drawing lane - left bottom 30 pixel to avoid drawing lanes on car dashboard shadow in camera
ploty = np.empty(0)
image_size = tuple()
def calibrate_camera():
    #function to calibrate camera using chessboard pictures
    import glob
    images= glob.glob('./camera_cal/calibration*.jpg')
    nx=9
    ny=6
    #we need 2 arrays to find destortion
    imagepoints = [] # 2d corner points detected by cv2 function
    objpoints = []  # 3d coordinates of chessboad like 0,0,0  0,1,0 etc
    objp = np.zeros((nx*ny,3),np.float32)
    objp [:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            imagepoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)
    return mtx,dist
def initialize():
    global first_frame
    global calibrate
    first_frame = True
    calibrate = calibrate_camera()
def get_M_Minv():
    global image_size
    #get M and M Inverse
    src = np.float32([[    (image_size[1]/4)-50,   image_size[0]-20],[ image_size[1]-80,   image_size[0]-20],[  ((image_size[1]/2)+140),   (2*image_size[0]/3)],[  (.4*image_size[1])-32,   (2*image_size[0]/3)]])
    dst = np.float32([[0,image_size[0]],[image_size[1],image_size[0]],[image_size[1],0],[0,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv
def mag_thresh(channel,gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
def get_perspective(img,M):
    #get perspective image
    img_size = image_size[1],image_size[0]
    image = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return image

def draw_pipeline(img, s_thresh=(150, 255), sx_thresh=(30, 100)):
    #gray channel threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    g_output = np.zeros_like(gray)
    g_output[(gray > s_thresh[0]) & (gray <= s_thresh[1])] = 1
    # Convert to HLS color space and separate the L and S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Sobel Magnitude  s channel
    smag_binary = mag_thresh(s_channel,gray, sobel_kernel=9, mag_thresh=(sx_thresh[0], sx_thresh[1]))
    # Combine gradient and colour treshold
    combined_binary = np.zeros_like(smag_binary)
    combined_binary[((g_output == 1) & (smag_binary == 1))] = 1
    return (combined_binary)
def fit_lines_first_time(img):
    #detect lane for first frame using histogram
    global first_frame
    global last_fit
    global first_fit
    global last_fitx
    global ploty
    global image_size
    
    binary_warped=np.copy(img)
    
    ploty = np.linspace(0, image_size[0]-1, image_size[0] )
    # Take a histogram of the image
    histogram = np.sum(binary_warped[:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            #if (np.int(np.mean(nonzerox[good_left_inds]))) > 300:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:      
            #if (np.int(np.mean(nonzerox[good_right_inds]))) < 900:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    last_fit = (left_fit,right_fit)
    first_fit = (left_fit,right_fit)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    first_frame = False
    last_fitx = (left_fitx,right_fitx)
    return left_fitx, right_fitx
def fit_lines_second_time_onwards(img):
    #detect lane from second frame onwards
    global first_frame
    global last_fit
    global first_fit
    global last_fitx
    global ploty
    global no_lane
    binary_warped=np.copy(img)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_fit,right_fit = first_fit[0],first_fit[1]
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each if every lane has enough pixels to form a lane
    if (leftx.size>50)and(lefty.size>50)and(rightx.size>50)and(righty.size>50):
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        last_fit = (left_fit,right_fit)
    else:
        left_fit, right_fit = last_fit[0], last_fit[1]
    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #if both the lanes starts converging at top or going in different direction then use the last reading
    
    left_diff = abs(left_fitx[0] - last_fitx[0][0])
    right_diff = abs(right_fitx[0] - last_fitx[1][0])
    #Check if the right and 
    mse_left = np.sqrt(np.mean(np.sum((left_fitx - last_fitx[0])**2)))
    mse_right = np.sqrt(np.mean(np.sum((right_fitx - last_fitx[1])**2)))
    #if (mse > (3*margin)):
        #no_lane = True
        #left_fitx,right_fitx = last_fitx[0], last_fitx[1]
    if (mse_left > (8*margin)):
        no_lane = True
        left_fitx = last_fitx[0]
    if (mse_right > (8*margin)):
        no_lane = True
        right_fitx = last_fitx[1]
    last_fitx = (left_fitx,right_fitx)
    return left_fitx, right_fitx
def draw_lines(binary_warped,undistort,left_fitx,right_fitx,Minv):
    #function to draw lines on the image
    global first_frame
    global last_fit
    global last_fitx
    global ploty
    global no_lane
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_size[1], image_size[0])) 
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    
    #Curvature
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    stcurv = 'Curvature: Left = ' + str(np.round(left_curverad,2)) + 'm, right = ' + str(np.round(right_curverad,2)) +'m' 
    # Change color if distance is more than 30 cm
    font = cv2.FONT_ITALIC    
    cv2.putText(result, stcurv, (30, 60), font, 1, (255,255,255), 3)
    car_center = int((left_fitx[690] + right_fitx[690])/2)
    img_center = result.shape[1]/2
    offset = abs(car_center - img_center)*xm_per_pix
    stoffset = 'Lane Deviation = ' + str(np.round(offset,2)) + ' m'
    cv2.putText(result, stoffset, (30, 110), font, 1, (255,255,255), 3)
    return result
def lane_detection(img):
    #main function to detect lanes and draw it on distorted image
    global calibrate
    global first_frame
    global last_fit
    global last_fitx
    global ploty
    global image_size
    image_size = img.shape
    M,Minv = get_M_Minv()
    mtx,dist = calibrate[0],calibrate[1]
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    pipeline = draw_pipeline(undistort)
    persimage = get_perspective(pipeline,M)
    img2=get_perspective(undistort,M)
    if first_frame:
        left_fitx, right_fitx = fit_lines_first_time(persimage)
    else:
        left_fitx, right_fitx = fit_lines_second_time_onwards(persimage)
    return draw_lines(persimage,undistort,left_fitx, right_fitx,Minv) 

