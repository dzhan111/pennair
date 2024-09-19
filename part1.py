import cv2
import numpy as np


image = cv2.imread('PennAir 2024 App Static.png')


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_yellow = np.array([22, 80, 80])
upper_yellow = np.array([35, 255, 255])

lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])

lower_green = np.array([45, 254, 250])  
upper_green = np.array([60, 255, 255]) 

lower_pink = np.array([140, 100, 100])
upper_pink = np.array([170, 255, 255])

#masks
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)


contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def draw_contours_and_centroids(contours, color):
    for contour in contours:

        cv2.drawContours(image, [contour], -1, (255, 255, 255), 3)
        

        M = cv2.moments(contour)
        
        if M["m00"] != 0:  
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0


        cv2.circle(image, (cX, cY), 7, color, -1)


draw_contours_and_centroids(contours_red, (255, 255, 255))
draw_contours_and_centroids(contours_yellow, (255, 255, 255))
draw_contours_and_centroids(contours_blue, (255, 255, 255))
draw_contours_and_centroids(contours_green, (255, 255, 255))
draw_contours_and_centroids(contours_pink, (255, 255, 255))

#display the image with border and centroids
cv2.imshow('Tracked Shapes with Centroids', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
