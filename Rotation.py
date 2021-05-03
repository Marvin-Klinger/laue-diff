#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marvin Klinger


######[Intro]##################################################################


This code was created in late 2020 to early 2021 to enhance the
Laue-Diffraction Machine at Augsburg University Chair of experiemental physics VI


This file expands on the "Basicfunctions.py" file and adds rotation tools.
For tutorials, updates and requests see:
    https://github.com/Marvin-Klinger/laue-diff

"""


"""
######[Imports]################################################################
"""

from Basicfunctions import motorClient, tC, laueClient, setExposure, maxFinder_large, showPlot, home
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
This function creates a 2D Array of angular pictures (height x width). Every
    point is scale degrees appart. Each picture is exposed for exposure_time.
    
    If the Array contains a probable candidate the sample will be aligned with
    this candidate.
    

"""
def simpleRotation(hight, width, scale, exposure_time, center_coordinates = (0,0)):
    
    #initial constants
    #minimum_time_for_dotsearch = 4999  #in ms of exposure
    #aperture_value = 1  #number written onto the X-Ray aperture
    
    offset_x = center_coordinates[0]
    offset_y = center_coordinates[1]
    
    dot_confidence_factor = 12.0
    stddev_confidence_factor = 0.3
    #mean_confidence_factor = 0.4
    area_confidence_factor = 1.0
    
    result_threshold = 0.9
    
    #initialie the containers
    result = np.zeros((width, hight))
    stddev = np.zeros(result.shape)
    avg = np.zeros(result.shape)
    area = np.zeros(result.shape)
    
    treffer = np.ones(result.shape, np.uint16)
    
    #timing calculations
    setExposure(exposure_time)
    duration = result.shape[0] * result.shape[1] * (exposure_time + 2000)
    print("Approximate duration: " + str(duration/1000) + "s, " + str(int(duration / 60000)+1) + "m")

    for col in range(result.shape[0]):
        
        #Move in x-axis
        xpos = (col - (result.shape[0]-1)/2.0)*scale + offset_x
        motorClient('R', 'X', xpos)
        
        for row in range(result.shape[1]):
            
            #Move in y-axis
            ypos = ((result.shape[1]-1)/2.0 - row)*scale + offset_y
            motorClient('R', 'Y', ypos)
        
            #Take the picture
            tC("192.168.1.10", 50000, "Snap\n")
            time.sleep(exposure_time/1000 + 0.5)
            image, rawMean, rawStddev, scaleMean, scaleStddev = laueClient("192.168.1.10",50000,"GetImage\n")
            #print("RawMean, RawStddev, sMean, sStddev: " + str(rawMean) + " " + str(rawStddev) + " " + str(scaleMean) + " " + str(scaleStddev) + " ")
            
            #Reduce dynamic Range to 8 bit
            gray = (image/256).astype('uint8')
            gamma = 0.8  
            
            #create gamma corrected image for maxFinder
            #lookUpTable = np.empty((1,256), np.uint8)
            #for i in range(256):
            #    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            gray_maxFinder = gray#cv2.LUT(gray, lookUpTable)
            
            
            #Find the Dots and save a copy of each analysis
            comment_string = "SimpleRotation(" + str(col) + "," + str(row) + ")"
            mD, sD, current_image = maxFinder_large(gray_maxFinder, True , False, comment_string,-3)
            
            #Get and save the image parameters
            result[col,row] = len(mD) + len(sD)
            stddev[col,row], _ = cv2.meanStdDev(gray)
            _, avg[col,row] = rawStddev, scaleMean
            area[col,row] = area_calculation(mD, sD)
            
            #set this tile to the "scanned but nothing found" status.
            #if something is found later on, this will be overwritten.
            treffer[col,row] = 0 #TODO maybe remove this statement
               
            #calculate the confidence for this tile
            
            treffer[col,row] += result[col,row] * dot_confidence_factor
            
            print(result[col,row] * dot_confidence_factor)
            
            treffer[col,row] += (255 - stddev[col,row]) * stddev_confidence_factor
            
            print((255 - stddev[col,row]) * stddev_confidence_factor)
            #treffer[col,row] += (255 - avg[col,row]) * mean_confidence_factor
            
            treffer[col,row] += area[col,row] * area_confidence_factor
            
            print(area[col,row] * area_confidence_factor)
            #print(str(treffer) + " = " + str(result[col,row] * dot_confidence_factor + (255 - stddev[col,row]) * stddev_confidence_factor + area[col,row] * area_confidence_factor))
            #print((255 - avg[col,row]) * mean_confidence_factor)
            
            #plot all the data and a current image
            #TODO determin wether to flip left <> right and top <> bottom
            plt.subplot(1,4,1)
            plt.title("Dot density:\n" + str(result[col,row])) 
            plt.imshow(np.flipud(np.fliplr(np.transpose(result))), cmap='inferno', vmin=0)
            #plt.xlim(((result.shape[1]-1)/2.0 - 0)*scale + offset_y, ((result.shape[1]-1)/2.0 - width)*scale + offset_y)
            #plt.ylim((0 - (result.shape[0]-1)/2.0)*scale + offset_x, (hight - (result.shape[0]-1)/2.0)*scale + offset_x)
            plt.subplot(1,4,2)
            plt.title("Stddev:\n" + str(int(stddev[col,row])))
            plt.imshow(np.flipud(np.fliplr(np.transpose(stddev))), cmap='inferno', vmin=0, vmax=255)
            plt.subplot(1,4,3)
            plt.title("Dot Area:\n" + str(area[col,row]))
            plt.imshow(np.flipud(np.fliplr(np.transpose(area))), cmap='inferno', vmin=0)
            plt.subplot(1,4,4)
            plt.title("Confidence:\n" + str(treffer[col,row]))
            plt.imshow(np.flipud(np.fliplr(np.transpose(treffer))), cmap='inferno', vmin=0)            
            plt.show()
            plt.title("Current image") 
            plt.imshow(current_image) 
            plt.show()
            
         
    #show general stats
    print("Done, result:")
    print(np.flipud(np.fliplr(np.transpose(treffer))))
    #print(np.average(result))    
    #print("Mean:")
    #print(avg)
    #print("stddev:")
    #print(stddev)
    
    #find the center of the contour
    #if there are values over 255 the entire matrix will be rescaled
    if(np.amax(treffer) > 255):
        treffer = (treffer * (255.0/np.amax(treffer))).astype('uint8')
    else:
        treffer = treffer.astype('uint8')
    
    plt.title("Trefferwahrscheinlichkeit:") 
    plt.imshow(np.flipud(np.fliplr(np.transpose(treffer))), cmap='inferno', vmin=0)
    plt.xlabel('x-Achse [index]')
    plt.ylabel('y-Achse [index]')
    plt.show()
    print(treffer)
    
    #showPlot(np.flipud(np.fliplr(treffer)), "Kontur")
    answer  = findCenter_rotation2(treffer, result_threshold)
    #([Data],[best Data, if avail])
    #TODO handle multiple equally likely maxima -> DONE
    print(answer)
    
    #TODO verify direction of travel
    
    print("y Position: "+  str(((result.shape[1]-1)/2.0 - answer[0])*scale + offset_y))
    ypos = ((result.shape[1]-1)/2.0 - answer[0])*scale + offset_y
    motorClient('R', 'Y', ypos)
    
    print("x Position: "+ str((answer[1] - (result.shape[0]-1)/2.0)*scale + offset_x))
    xpos = (answer[1] - (result.shape[0]-1)/2.0)*scale + offset_x
    motorClient('R', 'X', xpos)
    
    """if len(answer[1]) > 0:
        print("Maxlocation X: " + str((answer[1][0][0] - (result.shape[1]-1)/2.0)*scale))
        print("Maxlocation Y: " + str((answer[1][0][1] - (result.shape[0]-1)/2.0)*scale))
        
        #Move to the approximate center of the contour
        motorClient('M', 'Y', -(answer[1][0][0] - (result.shape[1]-1)/2.0)*scale)
        motorClient('M', 'X', (answer[1][0][1] - (result.shape[0]-1)/2.0)*scale)
        print("Moving the sample to the center of the contour... done")
        print("You may now take a long exposure image")
        return
    
    
    if len(answer[0]) > 0:
        #print(str(answer[0][0]) + " " + str(answer[0][1]))
        #print(answer[0][0])
        print("Maxlocation X: " + str((answer[0][0][0] - (result.shape[1]-1)/2.0)*scale))
        print("Maxlocation Y: " + str((answer[0][0][1] - (result.shape[0]-1)/2.0)*scale))
        
        #Move to the approximate center of the contour
        motorClient('M', 'Y', -(answer[0][0][0] - (result.shape[1]-1)/2.0)*scale)
        motorClient('M', 'X', (answer[0][0][1] - (result.shape[0]-1)/2.0)*scale)
        print("Moving the sample to the center of the contour... done")
        print("You may now take a long exposure image")
        return
    
    print("[Error] No contour has been identified!")
    home()
    """
    
"""
This function takes an array of confidence values as its input and returns the index
    of the highest confidence value. Since this center might not always be destinct,
    the average location of the highest confidence levels will be returned as well.
    
    Threshold 0.0 ...1.0 determines which tiles will be included into the contour
    for calculating the rough center
    
"""
    
def findCenter_rotation(confidence_matrix, threshold):
    highest_confidence = np.amax(confidence_matrix)
    most_confident_tiles = np.where(confidence_matrix == np.amax(confidence_matrix))
    
    thresh = cv2.threshold(confidence_matrix, int(highest_confidence * threshold), 255, cv2.THRESH_TOZERO)[1]
    print(thresh)
    contour_list = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    
    if(len(contour_list) > 0):
        largest_contour = contour_list[0]
    else:
        print("Error, no contours were found!")
    
    #find the largest contour
    for contour in contour_list:
        if cv2.contourArea(contour) > cv2.contourArea(largest_contour):
            largest_contour = contour
            
    print("largest contour")
    print(largest_contour)
    
    if(cv2.contourArea(largest_contour) > 1):
        print(cv2.contourArea(largest_contour))
        M = cv2.moments(largest_contour)
        cX = (M["m10"] / M["m00"])
        cY = (M["m01"] / M["m00"])
    else:
        cX = largest_contour[0][0][0]
        cY = largest_contour[0][0][1]
        
    contour_center_coordinates = (cX,cY)
    
    return most_confident_tiles, contour_center_coordinates


"""

"""

def findCenter_rotation2(confidence_matrix, threshold):
    highest_confidence = np.amax(confidence_matrix)
    most_confident_tiles = np.where(confidence_matrix == np.amax(confidence_matrix))
    
    print(len(most_confident_tiles))
    
    thresh = cv2.threshold(confidence_matrix, int(highest_confidence * threshold), 255, cv2.THRESH_TOZERO)[1]
    print(thresh)
    contour_list = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    
    
    if(len(contour_list) > 0):
        largest_contour = contour_list[0]
    else:
        print("Error, no contours were found!")
    
    #find the largest contour
    for contour in contour_list:
        if len(contour) > len(largest_contour):
            largest_contour = contour
            
    print("largest contour")
    print(largest_contour)
    
    if(cv2.contourArea(largest_contour) > 0):
        print(cv2.contourArea(largest_contour))
        M = cv2.moments(largest_contour)
        cX = (M["m10"] / M["m00"])
        cY = (M["m01"] / M["m00"])
        contour_center_coordinates = (cX,cY)
        return contour_center_coordinates
    else:
        #if no large contour has been found, this will check if there is a single
        #most probable candidate and pick it over the others.
        if(len(most_confident_tiles) == 2):
            print("Block reach")
            return(most_confident_tiles[1][0],most_confident_tiles[0][0])
          
        #No large contour (area > 0) has been identified and
        #no single most probable pixel has been found.
        #Will sum the equally likely candidates.
        
        counter = 0.0
        for element in most_confident_tiles[1]:
            cX += element
            cY += most_confident_tiles[0][counter]
            counter += 1
            
        return (cX / counter, cY / counter)
        
        #cX = largest_contour[0][0][0]
        #cY = largest_contour[0][0][1]
    
    
def area_calculation(largeDots, smallDots):
    total_area = 0
    for dot in largeDots:
        total_area += len(dot)
    
    for dot in smallDots:
        total_area += len(dot)
        
    print("Gesamtfläche: " + str(total_area))
    return total_area
    
    
    
    
    
    
    
    
    
    
    
    