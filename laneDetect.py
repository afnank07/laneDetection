import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def canny(image):
    ''' Step1: Convert Image to gray scale''' 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ''' Step2: Smoothen the image by Reducing noise. If this is not done than edge detection won't be accurate.
        We use gausian filter for this'''
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    ''' Step3: Use canny edge detection method. cv2.Canny(image, lower_threshold, higher_threshold) 
        usually both the threshold should be in 1:3 ratio'''
    canny = cv2.Canny(blur, 50, 150)
    return canny

''' Step4: is to define a triangular region that'll only include the right side 
    of the lane '''
def region_of_interest(image):
    # this will give us the length of the y-axis
    height = image.shape[0]
    # setting the points for the triangular region, 2 square brackets cause we are giving input for an array of polygons 
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    # this similar size complete black(0 image
    mask = np.zeros_like(image)
    # cv2.fillPoly(img, pts, color), creating a white(255) triangle with black background
    cv2.fillPoly(mask, polygons, 255)
    # performing bitwise AND operation to get specific area
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

''' Step6: overlap the lines onto the real image '''
def display_lines(image, lines):
    # create a black background
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:      # to convert a 2Darray to 1D array
            x1, y1, x2, y2 = line.reshape(4)
            # cv2.line(img, pt1, pt2, color, thickness)
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
    return line_image

''' Step7: all the blue lines will now be avgd to one line'''
def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # left lines have -ve slope cause as X increases Y decreases 
        if slope < 0:       
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
        left_fit_avg = np.average(left_fit, axis = 0) #axis = 0 to operate vertically
        right_fit_avg = np.average(right_fit, axis = 0)
        # to get x & y coordinates we pass it to this 
    try:
        left_line = make_coordinates(image, left_fit_avg)
        right_line = make_coordinates(image, right_fit_avg)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, '\n') #print error to console
        return None
        
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] #no.of rows would be the height 
    y2 = int(y1*(3/5))
    # x = (y-b)/m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

''' Applying the algorithm to Video '''
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    ''' Step5:Use Hough Transform to find out the straight lines formed by multiple points in our frame '''
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avg_slope_intercept(frame, lines)
    line_image = display_lines(frame, avg_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Output', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











