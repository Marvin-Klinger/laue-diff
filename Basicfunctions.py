#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marvin Klinger


######[Intro]##################################################################


This code was created in late 2020 to early 2021 to enhance the
Laue-Diffraction Machine at Augsburg University Chair of experiemental physics VI


Basics:
    To use this software you must include this file into your project .py
    
Example: (Take a 1000s exposure and save it)

    >import cv2
    >from Basicfunctions import longExposure
    >
    >image = longExposure(1000, True)
    >cv2.imwrite("./20210317_STO_1000s.tif", image)

As you can see, this makes working with the Laue machine easy.
For more complex examples see https://github.com/Marvin-Klinger/laue-diff
"""


"""
######[Imports]################################################################
"""
import time
import socket
import numpy as np
import cv2
from time import gmtime, strftime
from matplotlib import pyplot as plt



"""
######[Main Code]##############################################################

Functions in this section are optimized for direct interaction with users. Call
    them from your code.
"""

 

"""
reducePrecision(image)

Reduces a 16bit image to 8bit, rounding to the lower number. This is a lossy
    process.
"""

def reducePrecision(image):
    result = (image/256).astype('uint8')
    return result
 
    
    
"""
longExposure(expo_time_sec, getRawImage = False)

Allows taking long exposures, only limited by 16bit value size. Works by
    combining multiple 16s exposures and a final 1...16s exposure.
    
    expo_time_sec: exposure time in seconds
    
    getRawImage:
        True: return 16 bit image
        False: rescale the image to 8 bit
"""

def longExposure(expo_time_sec, getRawImage = False):
    number_of_exposures = int(expo_time_sec/16)
    print(number_of_exposures)
    remaining_exposure_time_ms = (expo_time_sec * 1000) - 16000 * number_of_exposures
    
    setExposure(16000)
    
    tC("192.168.1.10", 50000, "Snap\n")
    time.sleep(17)
    image,_,_ = laueClient("192.168.1.10",50000,"GetImage\n",True)
    
    final_picture = np.zeros(image.shape, np.uint16)
    final_picture = np.add(final_picture, image)
    
    for i in range(number_of_exposures - 1):
        tC("192.168.1.10", 50000, "Snap\n")
        time.sleep(17)
        image,_,_ = laueClient("192.168.1.10",50000,"GetImage\n", True)
        final_picture = np.add(final_picture, image)
    
    setExposure(int(remaining_exposure_time_ms))
    tC("192.168.1.10", 50000, "Snap\n")
    time.sleep((remaining_exposure_time_ms/1000.0) + 1)
    image,_,_ = laueClient("192.168.1.10",50000,"GetImage\n",True)
    final_picture = np.add(final_picture, image).astype('uint16')
    
    
    
    if getRawImage:
        return final_picture
    else:
        ny = 643
        nx = 975
        x_start = int(final_picture.shape[1]/2)-50-int(0.059*ny + 1)
        x_stop = int(final_picture.shape[1]/2)-int(0.059*ny + 1)
        y_start = int(final_picture.shape[0]/2)-50-int(0.059*ny + 1)
        y_stop = int(final_picture.shape[0]/2)-int(0.059*ny + 1)
        
        crop_img = final_picture[y_start:y_stop,x_start:x_stop]
        
        mean,_ = cv2.meanStdDev(crop_img)
        cv2.circle(final_picture, (int(nx/2) + 4,int(ny/2) + 4), int(0.054*ny+1), (mean[0][0],mean[0][0],mean[0][0]), -1)
        
        #Get the extreme luminance values and increase contrast so that the
        #   entire histogram is used (in order to not loose too much detail 
        #   when converting to 8-bit)
        minValue, maxValue,_,_ = cv2.minMaxLoc(final_picture)
        globalMin = np.ones(final_picture.shape)
        globalMin = globalMin * minValue
        final_picture = np.subtract(final_picture, globalMin)
        final_picture = final_picture * int(65535/(maxValue-minValue))
        return final_picture




"""
maxFinder(image, verbosity = False, saveThreshold = False, comment = "", adaptiveThresholdValue = -3, time = 0)

Identify the brightest places in an 8bit Image and return them as an array

    image: np.ndarray of uint8
    
    verbosity: bool
        True: show pictures during calculation
        False: perform calculation without additional display
        
    saveThreshold: save an additional Threshold image (B/W) containing the
        maxima
        
    comment: text included in the filename - usefull for naming
    
    adaptiveThresholdValue: Used for adaptive thresholding
    
    time: exposure time, will be added to the filename
"""

def maxFinder_large(image, verbosity = False, saveThreshold = False, comment = "", adaptiveThresholdValue = -3, time = 0):    
    #TODO:
    #> change adaptiveThreshold configuration according to mean / stdDev
 	
    #gray = cv2.addWeighted(gray, 1, gray, 0, 0)
    
    denoise = cv2.fastNlMeansDenoising(image,None,20,21,9)

    #Adaptive threshold
    tres = cv2.adaptiveThreshold(denoise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,61,adaptiveThresholdValue)
    
    #remove the outer areas - usually they are very noisy
    cv2.circle(tres, (491 ,325), 590, (0,0,0), 300)
    #TODO: check for correct orientation, probably reversed  
    nx = tres.shape[1]
    ny = tres.shape[0]
    cv2.circle(tres, (int(nx/2) + 4,int(ny/2) + 4), int(0.06*ny), (0,0,0), -1)
    
    #save the threshold image, only if selected
    #   especially for noisy images this provides a better output than the 
    #   extended multiplot
    if saveThreshold:
        saveImgTime(tres, "Threshold")

    # findcontours 
    cnts = cv2.findContours(tres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2] 

    cimage=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # filter by area 
    s1 = 8
    m = 60
    s2 = 2000
    smallDots = []
    mainDots = [] 
      
    for cnt in cnts:
        if s1<cv2.contourArea(cnt)<m: 
            smallDots.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cimage,(x,y),(x+w,y+h),(0,255,0),1)
            cimage = cv2.putText(cimage, str(cv2.contourArea(cnt)), (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA, False)
            cimage = cv2.putText(cimage, str(int(dotMean(image,cnt))), (x+w,y+h+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1, cv2.LINE_AA, False)
        if m<cv2.contourArea(cnt)<s2:
            mainDots.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cimage,(x,y),(x+w,y+h),(0,0,255),2)
            cimage = cv2.putText(cimage, str(cv2.contourArea(cnt)), (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA, False)
            cimage = cv2.putText(cimage, str(int(dotMean(image,cnt))), (x+w,y+h+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1, cv2.LINE_AA, False)
    
    #Average calculation
    xAvg = 0.0
    yAvg = 0.0
    for dot in mainDots:
        x,y,w,h = cv2.boundingRect(dot)
        xAvg += x
        yAvg += y
        
    if len(mainDots) > 0:
        xAvg = xAvg/len(mainDots) - image.shape[1]/2
        yAvg = yAvg/len(mainDots) - image.shape[0]/2
        
        #average vector draw:    
        cv2.arrowedLine(cimage, (int(image.shape[1]/2),int(image.shape[0]/2)), (int(image.shape[1]/2 +xAvg), int(image.shape[0]/2 +yAvg)), (255,0,0), 2)
    #else:
        #print("No mainDots found! Can not calculate average alignment!")
    
    #multiplot generation, only for verbose
    if verbosity:
        bar1 = np.concatenate((cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),cv2.cvtColor(denoise,cv2.COLOR_GRAY2BGR)),axis=1)
        bar2 = np.concatenate((cv2.cvtColor(tres,cv2.COLOR_GRAY2BGR),cimage),axis=1)
        bar1 = np.concatenate((bar1,bar2),axis=0)
        saveImgTime(bar1, comment, time)
        #bar1 = cv2.resize(bar1, (1500, 1000))
        #cv2.imshow('result',bar1)
    
    return mainDots, smallDots, cimage



"""
saveImgTime(image, [comment, [directory])

saves an image to disk with timestamp and exposure time.

    image: the image that will be saved
    
    comment: will be added to the filename
    
    time: exposure time, will be added to filename
    
    directory: where the image will be stored
    
    RAW: save 16bit (True) image or 8bit (False)
"""

def saveImgTime(image, comment = "", time = 0, directory = "./../10_Autosave/", RAW = False):
    if(time == 0):
        expo = laueClient("192.168.1.10",50000,"GetExposure\n")
        expo = expo.decode('utf-8')
        expo = expo.split(',')
        expo = expo[0]
        expo = expo[1:]
        if RAW:
            path = directory + strftime("%Y%m%d%H%M%S", gmtime()) + "_" + str(expo) + "ms_" + str(comment) + "_LaueAutosave.tif"
        else:
            path = directory + strftime("%Y%m%d%H%M%S", gmtime()) + "_" + str(expo) + "ms_" + str(comment) + "_LaueAutosave.bmp"
    else:
        if RAW:
            path = directory + strftime("%Y%m%d%H%M%S", gmtime()) + "_" + str(time) + "ms_" + str(comment) + "_LaueAutosave.tif"
        else:
            path = directory + strftime("%Y%m%d%H%M%S", gmtime()) + "_" + str(time) + "ms_" + str(comment) + "_LaueAutosave.bmp"
    
    cv2.imwrite(path, image)
    
    

    
"""
motorClient(arg, direction, value)

sends a command over the network to the motor server.
    arg:
        "M" : Move
        "S" : Shutdown
        "Q" : Get Position
        "R" : Rotate
        "C" : Kill the server
    
    direction:
        "X"
        "Y"
    
    value:
        in mm

"""    

def motorClient(arg, direction, value):
    message = arg + ";" + direction + ";" + str(value) + ";"
    host = "192.168.1.30" # socket server IPv4 address
    port = 40000  # socket server port number
    #print("socket connecting")
    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    client_socket.send(message.encode())  # send message
    data = client_socket.recv(1024).decode()  # receive response
    client_socket.close()  # close the connection
    #print("socket closed")
    #print('Received from server: ' + data)  # show in terminal
    ans = data.split(';')
    print(ans)





"""
simpleRaster_v2(hight, width, scale, exposure_time, coordinates = (0,0))

This function moves the sample in a xy raster shape and takes images. If a
    crystal is identified, the sample will be centered by the end of the run.
    
    hight: vertical resolution
        
    width: horizontal resolution
        
    scale: distance between measurements 
    
    exposure_time: exposure time for each sample
        
    coordinates: offset, useful for susequent runs, homing in on the target
"""   

def simpleRaster_v2(hight, width, scale, exposure_time, coordinates = (0,0)):
    #TODO: add offset
    #initial constants
    minimum_time_for_dotsearch = 4999  #in ms of exposure
    aperture_value = 2  #number written onto the X-Ray aperture
    #exposure_time = 5000
    
    #calibration array [stage][aperture_value] = data
    stddev_comparison_array = np.array([[175, 145, 105, 0],
                                        [170, 130, 110, 0],
                                        [162, 120, 90, 0],
                                        [0, 0, 0, 0]])
    
    #similar to stddev_comparison_array but for long exposures
    sddev_longexposure_value = np.array([190,185,190,100])
    
    print(stddev_comparison_array[2][aperture_value - 1])
    offset = (coordinates[1], -coordinates[0])
    
    #initialie the containers
    result = np.zeros((hight, width))
    stddev = np.zeros(result.shape)
    avg = np.zeros(result.shape)
    treffer = np.ones(result.shape, np.uint8)
    
    #timing calculations
    if exposure_time <= 16000:
        setExposure(exposure_time)
    
    duration = result.shape[0] * result.shape[1] * (exposure_time + 2000)
    print("Approximate duration: " + str(duration/1000) + "s, " + str(int(duration / 60000)+1) + "m")

    for col in range(result.shape[0]):
        
        #Move in x-axis
        #print(str((col - (result.shape[0]-1)/2.0)*scale) + str(offset[0]) + " = " + str((col - (result.shape[0]-1)/2.0)*scale + offset[0]))
        print("move x:")
        xpos = (col - (result.shape[0]-1)/2.0)*scale + offset[0]
        motorClient('M', 'X', xpos)
        
        for row in range(result.shape[1]):
            
            #Move in y-axis
            print("move y:")
            ypos = ((result.shape[1]-1)/2.0 - row)*scale + offset[1]
            motorClient('M', 'Y', ypos)
        
            #Take the picture
            if exposure_time > 16000:
                image = longExposure(exposure_time/1000)
                scaleMean, scaleStddev = cv2.meanStdDev(image)
                
            else:
                tC("192.168.1.10", 50000, "Snap\n")
                time.sleep(exposure_time/1000 + 0.5)
                image, rawMean, rawStddev, scaleMean, scaleStddev = laueClient("192.168.1.10",50000,"GetImage\n")
                #print("RawMean, RawStddev, sMean, sStddev: " + str(rawMean) + " " + str(rawStddev) + " " + str(scaleMean) + " " + str(scaleStddev) + " ")
            
            #Reduce dynamic Range to 8 bit
            gray = (image/256).astype('uint8')
            
            #Find the Dots and save a copy of each analysis
            comment_string = "SimpleRaster(" + str(col) + "," + str(row) + ")"
            mD, sD, current_image = maxFinder_large(gray, True , False, comment_string,-3, int(exposure_time/1000))
            
            #Get and save the image parameters
            result[col,row] = len(mD) + len(sD)
            stddev[col,row], _ = cv2.meanStdDev(gray)
            avg[col,row] = scaleMean
            
            #set this tile to the "scanned but nothing found" status.
            #if something is found later on, this will be overwritten.
            treffer[col,row] = 0
            
            #Debug output for setting the apropriate sddev levels
            print("Bright: " + str(stddev[col,row]))
            
            if exposure_time <= 5000:
                if stddev[col,row] < stddev_comparison_array[0][aperture_value - 1]:
                    treffer[col,row] = 2
            elif exposure_time <= 10000:
                if stddev[col,row] < stddev_comparison_array[1][aperture_value - 1]: #Maximalwert für TiO auf Alu: 160
                    treffer[col,row] = 2
            elif exposure_time <= 16000:
                if stddev[col,row] < stddev_comparison_array[2][aperture_value - 1]:
                    treffer[col,row] = 2
            else:
                if stddev[col,row] < sddev_longexposure_value[aperture_value - 1]:
                    treffer[col,row] = 2
                    
            #Draw the current progress for the user
            showPlotImage(treffer, "Scan" , current_image, "Aktuelles Bild")
      
    #if the exposure time is reasonably long, use the dot-search to determine the center of the picture
    if exposure_time > minimum_time_for_dotsearch:
        #print("Punktsucher:")
        #print(result)
        counter = 0.0;
        avgDotnumber = 0.0;
        
        for col in range(result.shape[0]):
            for row in range(result.shape[1]):
                #if  stddev[col,row] > 160:
                if treffer[col,row] != 2:
                    result[col,row] = 0
                else:
                    counter += 1
                    avgDotnumber += result[col,row]
                    
        if counter > 0:
            avgDotnumber = avgDotnumber / counter
            #print("Average Dotnumber in low stdev Areas: " + str(avgDotnumber))
                    
        for col in range(result.shape[0]):
            for row in range(result.shape[1]):
                #if result[col,row] < avgDotnumber and result[col,row] > 0:
                if result[col,row] > avgDotnumber:
                    treffer[col,row] = 3
    else:
        print("Short exposure time, only using stddev. This can lead to false negatives!")
    
    #show general stats
    print("Done, result:")
    print(treffer)
    #print(np.average(result))    
    #print("Mean:")
    #print(avg)
    #print("stddev:")
    #print(stddev)
    
    #find the center of the contour
    print("Suche das Zentrum der Konturen:")
    showPlot(treffer, "Kontur")
    answer = findCenter(treffer)
    
    #print(answer)
    #print(len(answer[0]))
    #print(len(answer[1]))
    #print(result.shape)
    
    if len(answer[1]) > 0:
        print("Maxlocation X: " + str((answer[1][0][0] - (result.shape[1]-1)/2.0)*scale + offset[1]))
        print("Maxlocation Y: " + str((answer[1][0][1] - (result.shape[0]-1)/2.0)*scale + offset[0]))
        
        #Move to the approximate center of the contour
        motorClient('M', 'Y', -((answer[1][0][0] - (result.shape[1]-1)/2.0)*scale) + offset[1])
        motorClient('M', 'X', (answer[1][0][1] - (result.shape[0]-1)/2.0)*scale + offset[0])
        print("Moving the sample to the center of the contour... done")
        print("You may now take a long exposure image")
        return
    
    
    if len(answer[0]) > 0:
        #print(str(answer[0][0]) + " " + str(answer[0][1]))
        #print(answer[0][0])
        print("Maxlocation X: " + str((answer[0][0][0] - (result.shape[1]-1)/2.0)*scale + offset[1]))
        print("Maxlocation Y: " + str((answer[0][0][1] - (result.shape[0]-1)/2.0)*scale + offset[0]))
        
        #Move to the approximate center of the contour
        motorClient('M', 'Y', -((answer[0][0][0] - (result.shape[1]-1)/2.0)*scale) + offset[1])
        motorClient('M', 'X', (answer[0][0][1] - (result.shape[0]-1)/2.0)*scale + offset[0])
        print("Moving the sample to the center of the contour... done")
        print("You may now take a long exposure image")
        return
    
    print("[Error] No contour has been identified!")
    home()
    

    
"""
Horizontal pan function, mostly useful for taking videos

"""    

def Pan(samples, scaling, offset, direction='X', save = False):
    result = [0 for i in range(samples)]
    scaleMean = [0 for i in range(samples)]
    rawStddev = [0 for i in range(samples)]
    expo = getExpoSec("192.168.1.10",50000)
    for index in range(samples):
        winkelf = (index - offset)*scaling
        
        print("Moving to " + str(winkelf)+ "...")
        motorClient('R', direction, winkelf)
        
        tC("192.168.1.10", 50000, "Snap\n")
        time.sleep(expo + 0.5)
        image,rawMean, rawStddev[index],scaleMean[index] , scaleStddev = laueClient("192.168.1.10",50000,"GetImage\n")
        #print("RawMean, RawStddev, sMean, sStddev: " + str(rawMean) + " " + str(rawStddev[index]) + " " + str(scaleMean[index]) + " " + str(scaleStddev) + " ")
        gray = (image/256).astype('uint8')
    
        #print("Mean, Stdev:")
        #print(cv2.meanStdDev(gray))
        mD, sD = maxFinder(gray, save, True, str(winkelf), -3)
        result[index] = len(sD)
    
    return result, scaleMean, rawStddev



def verticalPan(samples, scaling, offset, save = False):
    result = [0 for i in range(samples)]
    expo = getExpoSec("192.168.1.10",50000)
    for index in range(samples):
        winkelf = (index - offset)*scaling
        
        print("Moving to " + str(winkelf)+ "...")
        motorClient('R', 'Y', winkelf)
        
        tC("192.168.1.10", 50000, "Snap\n")
        time.sleep(expo + 0.5)
        
        image, rawMean = laueClient("192.168.1.10",50000,"GetImage\n")
        gray = (image/256).astype('uint8')
    
        #print("Mean, Stdev:")
        #print(cv2.meanStdDev(gray))
        mD, sD = maxFinder(gray, save, save, -3)
        sD = np.concatenate(mD, sD)
        result[index] = len(sD)
    
    return result

"""

"""  

def home():
    motorClient('M', 'X', 0.0)
    motorClient('M', 'Y', 0.0)
    motorClient('R', 'X', 0.0)
    motorClient('R', 'Y', 0.0)
    
    
    


def savePicture():
    image,rawMean,rawStddev,scaleMean,scaleStddev = laueClient("192.168.1.10",50000,"GetImage\n")
    #print("RawMean, RawStddev, sMean, sStddev: " + str(rawMean) + " " + str(rawStddev[index]) + " " + str(scaleMean[index]) + " " + str(scaleStddev) + " ")
    gray = (image/256).astype('uint8')
    mD, sD = maxFinder(gray, True, False, "single_save", -3)
    



    
    


    
    
def bracketing(interval_minutes, number_of_pictures, getRawImage = False):
    for picture_id in range(number_of_pictures):
        _ = tC("192.168.1.10", 50000, "Snap\n")
        print("taking picture...")
        time.sleep(interval_minutes * 60 + 0.5)
        image,_, _,_ , _ = laueClient("192.168.1.10",50000,"GetImage\n", getRawImage)
        gray = (image/256).astype('uint8')
        print("saving picture...")
        saveImgTime(gray, "barcketing_id=" + str(picture_id))
        
def bracketing2(exposure_time_ms, interval_minutes, number_of_pictures, getRawImage = True):
    for picture_id in range(number_of_pictures):
        image = longExposure(int(exposure_time_ms/1000), getRawImage)
        saveImgTime(image, "barcketing_id=" + str(picture_id), int(exposure_time_ms/1000), getRawImage)
        

        
        
        
        
        
def client_program(message):
    host = "192.168.1.30" # socket server IPv4 address
    port = 40000  # socket server port number
    print("socket connecting")
    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    client_socket.send(message.encode())  # send message
    data = client_socket.recv(1024).decode()  # receive response
    client_socket.close()  # close the connection
    print("socket closed")
    print('Received from server: ' + data)  # show in terminal
    ans = data.split(';')
    print(ans)

def moveToPoint(coordinates = (0,0)):
    
    x = -coordinates[1]
    y = -coordinates[0]
    
    print(str(x))
    print(str(y))
    
    client_program("M;X;"+str(x)+";")
    client_program("M;Y;"+str(y)+";")
    


"""
#######[Old section]###########################################################

Code in this section has been superceeded by later versions but is beeing kept 
for compatibility.
"""









"""
SimpleRaster(hight, width, scale)

This function creates a simple 2D array of the sample, using the x/y linear
    drive. This is useful for identifying the shape of the sample.
    The answer is returned as a 2D array of confidence levels according to
    the exposure time.
    If the exposure time is sufficient, each individual maxima will be counted,
    otherwise a standard-deviation calculation will be performed.
    
    height: vertical resolution
    
    width: horizontal resolution
    
    scale: distance between two samples in mm (float)
    

"""

def simpleRaster(hight, width, scale):
    #initial constants
    minimum_time_for_dotsearch = 5000  #in ms of exposure
    aperture_value = 1  #number written onto the X-Ray aperture
    exposure_time = 16000
    
    #calibration array [stage][aperture_value] = data
    stddev_comparison_array = np.array([[150, 145, 105, 0],
                                        [160, 130, 110, 0],
                                        [166, 115, 90, 0],
                                        [0, 0, 0, 0]])
    
    #similar to stddev_comparison_array but for long exposures
    sddev_longexposure_value = np.array([160,160,160,160])
    
    stddev_comparison_array[aperture_value - 1][0]
    
    home()
    
    result = np.zeros((hight, width))
    stddev = np.zeros(result.shape)
    avg = np.zeros(result.shape)
    treffer = np.ones(result.shape, np.uint8)
    
    setExposure(exposure_time)
    duration = result.shape[0] * result.shape[1] * (exposure_time + 2000)
    print("Approximate duration: " + str(duration/1000) + "s, " + str(int(duration / 60000)) + "m")

    for col in range(result.shape[0]):
        
        #Move in x-axis
        xpos = (col - (result.shape[0]-1)/2.0)*scale
        motorClient('M', 'X', xpos)
        
        for row in range(result.shape[1]):
            
            #Move in y-axis
            ypos = ((result.shape[1]-1)/2.0 - row)*scale
            motorClient('M', 'Y', ypos)
        
            #Take the picture
            tC("192.168.1.10", 50000, "Snap\n")
            time.sleep(exposure_time/1000 + 0.5)
            image, rawMean, rawStddev, scaleMean, scaleStddev = laueClient("192.168.1.10",50000,"GetImage\n")
            #print("RawMean, RawStddev, sMean, sStddev: " + str(rawMean) + " " + str(rawStddev) + " " + str(scaleMean) + " " + str(scaleStddev) + " ")
            
            #Reduce dynamic Range to 8 bit
            gray = (image/256).astype('uint8')
            
            #Find the Dots and save a copy of each analysis
            comment_string = "Rotation(" + str(col) + "," + str(row) + ")"
            mD, sD, current_image = maxFinder(gray, True , False, comment_string,-3)
            
            #Get and save the image parameters
            result[col,row] = len(mD) + len(sD)
            stddev[col,row], _ = cv2.meanStdDev(gray)
            _, avg[col,row] = rawStddev, scaleMean
            
            #set this tile to the "scanned but nothing found" status.
            #if something is found later on, this will be overwritten.
            treffer[col,row] = 0
            
            #Debug output for setting the apropriate sddev levels
            print("Standardabweichung: " + str(stddev[col,row]))
            
            if exposure_time <= 5000:
                if stddev[col,row] < stddev_comparison_array[0][aperture_value - 1]:
                    treffer[col,row] = 2
            elif exposure_time <= 10000:
                if stddev[col,row] < stddev_comparison_array[1][aperture_value - 1]: #Maximalwert für TiO auf Alu: 160
                    treffer[col,row] = 2
            elif exposure_time <= 16000:
                if stddev[col,row] < stddev_comparison_array[2][aperture_value - 1]:
                    treffer[col,row] = 2
            else:
                if stddev[col,row] < sddev_longexposure_value[aperture_value - 1]:
                    treffer[col,row] = 2
                    
            #Draw the current progress for the user
            showPlotImage(treffer, "Scan" , current_image, "Aktuelles Bild")
      
    #if the exposure time is reasonably long, use the dot-search to determine the center of the picture
    if exposure_time > minimum_time_for_dotsearch:
        print("Punktsucher:")
        print(result)
        counter = 0.0;
        avgDotnumber = 0.0;
        
        for col in range(result.shape[0]):
            for row in range(result.shape[1]):
                #if  stddev[col,row] > 160:
                if treffer[col,row] != 2:
                    result[col,row] = 0
                else:
                    counter += 1
                    avgDotnumber += result[col,row]
                    
        if counter > 0:
            avgDotnumber = avgDotnumber / counter
            print("Average Dotnumber in low stdev Areas: " + str(avgDotnumber))
                    
        for col in range(result.shape[0]):
            for row in range(result.shape[1]):
                #if result[col,row] < avgDotnumber and result[col,row] > 0:
                if result[col,row] > avgDotnumber:
                    treffer[col,row] = 3
    else:
        print("Short exposure time, only using stddev. This can lead to false negatives!")
    
    print("Done, result:")
    print(treffer)
    print(np.average(result))    
    print("Mean:")
    print(avg)
    print("stddev:")
    print(stddev)
    
    print("Suche das Zentrum der Konturen:")
    showPlot(treffer, "Kontur")
    answer = findCenter(treffer)
    print(str(answer[0][0]) + " " + str(answer[0][1]))
    print("Maxlocation X: " + str((answer[0][0][0] - (result.shape[0]-1)/2.0)*scale))
    print("Maxlocation Y: " + str((answer[0][0][1] - (result.shape[1]-1)/2.0)*scale))
    
    #Move to the approximate center of the contour
    motorClient('M', 'X', (answer[0][0][0] - (result.shape[0]-1)/2.0)*scale)
    motorClient('M', 'Y', (answer[0][0][1] - (result.shape[1]-1)/2.0)*scale)




"""
maxFinder(image, verbosity)

Identify the brightest places in an 8bit Image and return them as an array

    image: np.ndarray of uint8
    
    verbisity: bool
    >true: show pictures during calculation
    >false: perform calculation without additional display
"""
    
def maxFinder(image, verbosity = False, saveThreshold = False, comment = "", adaptiveThresholdValue = -3):    
    #TODO:
    #> change adaptiveThreshold configuration according to mean / stdDev
 	
    #gray = cv2.addWeighted(gray, 1, gray, 0, 0)
    
    denoise = cv2.fastNlMeansDenoising(image,None,21,2,11)

    #Adaptive threshold
    tres = cv2.adaptiveThreshold(denoise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,61,adaptiveThresholdValue)
    
    #remove the outer areas - usually they are very noisy
    cv2.circle(tres, (491 ,325), 590, (0,0,0), 300)
    
    #save the threshold image, only if selected
    #   especially for noisy images this provides a better output than the 
    #   extended multiplot
    if saveThreshold:
        saveImgTime(tres, "Threshold")

    # findcontours 
    cnts = cv2.findContours(tres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2] 

    cimage=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    # filter by area 
    s1 = 2
    m = 60
    s2 = 2000
    smallDots = []
    mainDots = [] 
      
    for cnt in cnts:
        if s1<cv2.contourArea(cnt)<m: 
            smallDots.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cimage,(x,y),(x+w,y+h),(0,255,0),1)
            cimage = cv2.putText(cimage, str(cv2.contourArea(cnt)), (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA, False)
            cimage = cv2.putText(cimage, str(int(dotMean(image,cnt))), (x+w,y+h+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1, cv2.LINE_AA, False)
        if m<cv2.contourArea(cnt)<s2:
            mainDots.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(cimage,(x,y),(x+w,y+h),(0,0,255),2)
            cimage = cv2.putText(cimage, str(cv2.contourArea(cnt)), (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA, False)
            cimage = cv2.putText(cimage, str(int(dotMean(image,cnt))), (x+w,y+h+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 1, cv2.LINE_AA, False)
    """
    #Average calculation
    xAvg = 0.0
    yAvg = 0.0
    for dot in mainDots:
        x,y,w,h = cv2.boundingRect(dot)
        xAvg += x
        yAvg += y
        
    if len(mainDots) > 0:
        xAvg = xAvg/len(mainDots) - image.shape[1]/2
        yAvg = yAvg/len(mainDots) - image.shape[0]/2
        
        #average vector draw:    
        cv2.arrowedLine(cimage, (int(image.shape[1]/2),int(image.shape[0]/2)), (int(image.shape[1]/2 +xAvg), int(image.shape[0]/2 +yAvg)), (255,0,0), 2)
    #else:
        #print("No mainDots found! Can not calculate average alignment!")
    """
    #multiplot generation, only for verbose
    if verbosity:
        bar1 = np.concatenate((cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),cv2.cvtColor(denoise,cv2.COLOR_GRAY2BGR)),axis=1)
        bar2 = np.concatenate((cv2.cvtColor(tres,cv2.COLOR_GRAY2BGR),cimage),axis=1)
        bar1 = np.concatenate((bar1,bar2),axis=0)
        saveImgTime(bar1, comment)
        #bar1 = cv2.resize(bar1, (1500, 1000))
        #cv2.imshow('result',bar1)
    
    return mainDots, smallDots, cimage


"""
testPanMaxintens()

A test for comparining diffenet metrics for orienting the sample.
"""

def testPanMaxintens():
    dotConfidence = 1
    BrightConfidence = 1
    StddevConfidence = 1
    
    print("begin")    
    dotlist, meanlist, stddevlist = Pan(10, 0.5, 0, 'X', True)
    confidence = np.zeros(len(dotlist))
    print("Maximum of dots:")
    print(dotlist)
    result = np.where(dotlist == np.amax(dotlist))
    
    for max in result[0]:
        confidence[max] += dotConfidence
    
    print(result[0])
    print("Maximum of Mean-Brightness:")
    print(meanlist)
    result2 = np.where(meanlist == np.amax(meanlist))
    
    for max in result2[0]:
        confidence[max] += BrightConfidence
    
    print(result2[0])
    print("Minimum of Stddev:")
    print(stddevlist)
    result3 = np.where(stddevlist == np.amin(stddevlist))
    
    for max in result3[0]:
        confidence[max] += StddevConfidence
    
    print(result3[0])
    print("confidence")
    print(confidence)
    index = np.where(confidence == np.amax(confidence))
    winkel = (index[0][0] + 0)*0.5
    print(winkel)
    
    home()













"""
#######[Service section]#######################################################

These declarations are not for direct use but are called by other functions
"""


"""
laueClient(ip, port, message, getRawImage = False, only16I = True)

An advanced client for the NXT Laue Camera. It can run the <message> command 
and will relay the answer back and return it.
    If the answer is picture it will be returned as a numpy ndarray.
    The picture must be encoded as 16 bit integer!
    For non-picture answers, the byte-String is directly returned.
    If <getRawImage> is True the raw 16bit image will be returned. Otherwise
    the image will be rescaled.
    If <only16I> is true the client will ignore 8bit images.
    
    ip: String 
        > pattern: "[0...255].[0...255].[0...255].[0...255]"
        
    port: Integer 
        > default: 50000
        
    message: String
        > see manufacturer doc
"""

def laueClient(ip, port, message, getRawImage = False, only16I = True):
    #print("started")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #print("socket ok")
    sock.connect((ip, port))
    #print("connection ok")
    answer = ""
    sock.send(str.encode(message))
    #print("message sent")
    
    if message.lower() in ["getimage\n","getnewimage\n"]:
        #print("Data transfer detected!")
        data = b''
        dlen = 0;
        
        #TODO correct inital recv call
        if only16I:
            rep = sock.recv(20)
        else:
            rep = sock.recv(28)
        
        #print(rep)
        #print(len(rep))
        rep = rep.decode('utf-8')
        #print(rep)
        ans = rep.split(';')
        if len(ans)==4:
            #print("8-bit Imaging")
            mode,nx,ny,data_len = ans
        elif ans[0]=="I" and ans[1]=="16":
            #print("16-bit Imaging")
            nx,ny,data_len = ans[2],ans[3],ans[4]
            
            for i in range(len(data_len)):
                if not (data_len[i].isdigit()):
                    dlen = data_len[:i]
                    data = data_len[i:].encode('utf-8')
                    #print("leftover data:")
                    #print(data)
                    break
                    
        else:
            print("[Error] answer out of bounds")
            return False
        
        nx = int(nx)
        ny = int(ny)
        if int(dlen) > 0:
            data_len = int(dlen)
        else:
            data_len = int(data_len)

        #Read the data from the socket until data_len bytes have transferred
        while 1:
            rep = sock.recv(10240)
            data = b''.join([data,rep])
            if len(data)>=data_len:
                break
            
        sock.close()
        #print("{} kB transferred".format(len(data)/1000))
        
        #Prepare the image as a np.array
        img = np.frombuffer(data, np.uint16).reshape(ny, nx)
        
        #Preserve the mean of the original picture. This might be usefull when
        #   determining sample alignment. This will be a 16bit value!
        rawMean,rawStddev = cv2.meanStdDev(img)
        
        #No modifications will be made
        if(getRawImage):
            return img, rawMean[0][0], rawStddev[0][0]
        
        #Blank out the center circle. For this the mean center 
        #   brightness is calculated from a 50x50 square
        #crop_img = img[int(img.shape[0]/2)-50:int(img.shape[0]/2), int(img.shape[1]/2)-50:int(img.shape[1]/2)]
        x_start = int(img.shape[1]/2)-50-int(0.059*ny + 1)
        x_stop = int(img.shape[1]/2)-int(0.059*ny + 1)
        y_start = int(img.shape[0]/2)-50-int(0.059*ny + 1)
        y_stop = int(img.shape[0]/2)-int(0.059*ny + 1)
        
        crop_img = img[y_start:y_stop,x_start:x_stop]
        
        mean,_ = cv2.meanStdDev(crop_img)
        cv2.circle(img, (int(nx/2) + 4,int(ny/2) + 4), int(0.054*ny+1), (mean[0][0],mean[0][0],mean[0][0]), -1)
        
        #Get the extreme luminance values and increase contrast so that the
        #   entire histogram is used (in order to not loose too much detail 
        #   when converting to 8-bit)
        minValue, maxValue,_,_ = cv2.minMaxLoc(img)
        globalMin = np.ones(img.shape)
        globalMin = globalMin * minValue
        img = np.subtract(img, globalMin)
        img = img * int(65535/(maxValue-minValue))
        
        scaleMean, scaleStddev = cv2.meanStdDev(img)
        
        return img, rawMean[0][0], rawStddev[0][0], scaleMean[0][0], scaleStddev[0][0]
        
    else:
        answer = sock.recv(4096)
        sock.close()
        return answer




"""
markDot(image, dot)

marks a Dot with a colored rectangle frame surrounding the contour.

    image: the contour will be marked by a rectangle in this image.
    
    dot: countour to be marked
"""

def markDot(image, dot):
        x,y,w,h = cv2.boundingRect(dot)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2)


"""
showPlotImage(array, titel1, image, titel2)

Tool for plotting subplots
"""

def showPlotImage(array, titel1, image, titel2):
    plt.subplot(2,2,1)
    plt.title(titel1) 
    plt.imshow(array, cmap='jet', vmin=0, vmax=3)
    plt.subplot(2,2,2)
    plt.title(titel2)
    plt.imshow(image) 
    plt.show()


"""
showPlot(array, titel)

Tool for plotting arrays
"""

def showPlot(array, titel):
    plt.title(titel) 
    plt.imshow(array, cmap='jet', vmin=0, vmax=3) 
    plt.show()



"""
getExpoSec(ip, port)

Returns the exposure time currently set on the Laue-PC
"""
    
def getExpoSec(ip, port):
    expo = laueClient(ip, port,"GetExposure\n")
    expo = expo.decode('utf-8')
    expo = expo.split(',')
    expo = expo[0]
    expo = expo[1:]
    expo = int(expo)
    return expo


"""
brightnessAnalysis(multiplyer, size)

An inefficient way of taking pictures with increasing exposure times. Not
    recomended for new projects.
    
    multiplyer: exposure time in seconds for first image. The next image will
        take 2x as long, the next 3x and so on.
        
    size: number of pictures
"""

def brightnessAnalysis(multiplyer, size):
    for index in range(size):
        image = longExposure(int(multiplyer*(index+1)), True)
        saveImgTime(image, "PVC2_barcketing_id=" + str(index),multiplyer*(index+1), "./../10_Autosave/20210304_B3_Serienaufnahme/", True)
#        cv2.imwrite("./../10_Autosave/"+ str(index) + ".tiff", image)


"""
setExposure(exposure_time_milliseconds)

Sets the exposure time on the Laue-PC. Expects value in ms.
"""

def setExposure(exposure_time_milliseconds):
    laueClient("192.168.1.10",50000,"SetExpoMS;" + str(exposure_time_milliseconds) + "\n")
    time.sleep(1)
    if(getExpoSec("192.168.1.10",50000) != exposure_time_milliseconds):
        print("Error while setting exposure time! Unable to set correct exposure time.")


"""
draw_image_histogram(image, channels, color='k')

Draws a simple Histogram of the supplied image channels in the specified color.
    For use with color-pictures and 16bit depth.
"""

def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [65536], [0, 65536])
    plt.plot(hist, color=color)
    plt.xlim([0, 65536])
    
    
    
"""
draw_image_histogram(image, channels, color='k')

Draws a simple Histogram of the supplied image channels in the specified color.
    For use with color-pictures and 8bit depth.
"""

def draw_image_histogram_8(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [255], [0, 255])
    plt.plot(hist, color=color)
    plt.xlim([0, 255])



"""
show_grayscale_histogram(grayscale_image)

Draws a simple Histogram of the supplied image in black.
    For use with 8bit B/W-pictures.
"""

def show_grayscale_histogram(grayscale_image):
    draw_image_histogram(grayscale_image, [0])
    plt.show()


    
"""
show_grayscale_histogram(grayscale_image)

Draws a simple Histogram of the supplied image in black.
    For use with 16bit B/W-pictures.
"""

def show_grayscale_histogram_8(grayscale_image):
    draw_image_histogram_8(grayscale_image, [0])
    plt.show()

"""

This function takes confidence matricies with values from 0-3 and isolates the
    areas of interest i.e. the location of the crystal in the scan.
    Additionally the largest contour (by circumference) is determined and
    returned first.
    To better distinguish between lvl 2 and lvl 3 features the cumputations are
    done without any filters first. Any confidence will be regarded equally.
    After that a second pass is executed and only the confidence 3 tiles are
    taken into account.

"""
#TODO change contour Area to contour length 
def findCenter(confidence):
    center_lvl2 = []
    center_lvl3 = []
    cnts = cv2.findContours(confidence, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    i = 0
    
    #find the largest contur and save it as the first in the stack
    if(len(cnts) > 0):
        largest_cnt = cnts[0]
        for test_cnt in cnts:
            if len(test_cnt) > len(largest_cnt):
                largest_cnt = test_cnt
        
        #before appending the contour to the list the size must be verified.
        #   The largest Contour should be larger than 1 tile
        if(cv2.contourArea(largest_cnt) >= 1):
            M = cv2.moments(largest_cnt)
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            center_lvl2.append((cX,cY))
            print("A large contour (lvl 2) has been identified: (" + str(cX) + " " + str(cY) + ")")
        else:
            print("no contour larger than 1 has been found!")
        
    #now add the following contures to the list unordered. This will add the previously
    #   determined main contour aswell
    for c in cnts:
        if(cv2.contourArea(c) > 1):
            M = cv2.moments(c)
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
        else:
            cX = c[0][0][0]
            cY = c[0][0][1]
        
        center_lvl2.append((cX,cY))
        i += 1


    #recompute the contures for a different threshold. Only the probable (3)
    #   tiles will be used.
    thresh = cv2.threshold(confidence, 2, 255, cv2.THRESH_TOZERO)[1]
    cnts3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #find the largest contur - this is largely a duplicate from the first block
    if(len(cnts3) > 0):
        largest_cnt = cnts3[0]
        for test_cnt in cnts3:
            if cv2.contourArea(test_cnt) > cv2.contourArea(largest_cnt):
                largest_cnt = test_cnt
        
        #before appending the contour to the list the size must be verified.
        #   The largest Contour should be larger than 1 tile
        if(cv2.contourArea(largest_cnt) >= 1):
            M = cv2.moments(largest_cnt)
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            center_lvl3.append((cX,cY))
            print("The largest contour (lvl 3) has been identified: (" + str(cX) + " " + str(cY) + ")")
        else:
            print("no contour larger than 1 has been found!")
    
    #reset the appending-index
    i = 0
    for c in cnts3:
        if(cv2.contourArea(c) > 1):
            M = cv2.moments(c)
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
        else:
            cX = c[0][0][0]
            cY = c[0][0][1]
        
        center_lvl3.append((cX,cY))
        i += 1
            
    
    return center_lvl2, center_lvl3


"""
tC(ip, port, message)

A simple client for the NXT Laue Camera. It can run the <message> and will
    relay the answer back and return it.
    The maximum answer size is 4096 bits, which allows all <message> commands
    except those that return an image.
    
    ip: String 
        > pattern: "[0...255].[0...255].[0...255].[0...255]"
        
    port: Integer 
        > default: 50000
        
    message: String
        > see manufacturer doc
"""

def tC(ip, port, message):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((ip,port))
    answer = ""
    soc.send(str.encode(message))
    answer = soc.recv(4096)
    soc.close
    return answer

"""
dotMean(dot)

Takes a contour and calculates the mean value inside a bounding rectangle of 
    the contour.
"""

def dotMean(image, dot):
    x,y,w,h = cv2.boundingRect(dot)
    if w > 0 and h > 0:
        #print(x+w)
        #print(y+h)
        rect = image[y:y+h,x:x+w]
        mean,_ = cv2.meanStdDev(rect)
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        return mean[0][0]
    else:
        return 0



"""
maxIntensDot(image, dots)

Finds and returns the median-brightest dot in an image. 
"""

def maxIntensDot(image, dots):
    if len(dots) == 0:
        return 0
    
    maxintens = dots[0]
    for dot in dots:    
        if dotMean(image, dot) > dotMean(image, maxintens):
            maxintens = dot
    
    return maxintens