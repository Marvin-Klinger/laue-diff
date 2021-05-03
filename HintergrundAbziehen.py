#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marvin Klinger


######[Intro]##################################################################


This code was created in late 2020 to early 2021 to enhance the
Laue-Diffraction Machine at Augsburg University Chair of experiemental physics VI


Some pictures will contain lots of background brightness (mostly in the center).
These functions try to estimate the global background and remove it.

Example:
    > img = cv2.imread("./Entfernung_HG/20210317_STO_1000s.tif", -1)
    > img = removeBackgroundFit2(img)
    
For more complex examples see https://github.com/Marvin-Klinger/laue-diff
"""


"""
######[Imports]################################################################
"""
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import cmath
import math


def fitfunction(x, I, c):
    return I/(((487.5-x)*7.75/487.5)**2 + 4.02**2) + c

def fitfunc2(x,I, c, p, a):
    nenner = ((a*x-p)**2 + (4.02)**2)**(0.5)
    return I*np.arctanh((p-a*x)/nenner) + c

def zweidim(x, d, I, c, a):
    #return I*(1/40.2)*np.arctan((d-x)/40.2) - I*(1/40.2)*np.arctan((-d-x)/40.2) + c
    #return I * np.arctan(d - a (x - 487)) - I*np.arctan(-d - a (x - 487)) + c
    return I*np.arctan((d-a*(x-487.5))/(40.2)) - I*np.arctan((-d-a*(x-487.5))/(40.2)) + c

def removeBackgroundFit(img, filepath = "./test.jpg", verbose = True):
    x = np.arange(975)
    x2 = x[0:440]
    b = np.zeros((img.shape[0],), dtype=int)
    b[int(img.shape[0] / 2) + 5]=1
    mod = np.dot(b,img)
    mod2 = mod[0:440]
    fit_params, pcov = scipy.optimize.curve_fit(fitfunc2, x2, mod2)
    
    if verbose:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(x, mod, label=alias[index], s=1, alpha=1)
        residuals = mod2- fitfunc2(x2, *fit_params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mod2-np.mean(mod2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        y_fit = fitfunc2(x, *fit_params)
        ax.plot(x, y_fit, label=alias[index]+' fit (' + str(round(r_squared,3)) + ')', linewidth="1", color = 'black')
        ax.set(xlabel='Horizontale Position [Pixel]', ylabel='Wert [W.E.]')
        ax.grid()
        ax.set_xlim([0,500])
        plt.legend(loc="upper left")
            
    
    mat = np.zeros(img.shape, dtype=int)
    dist = np.zeros(img.shape, dtype=int)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            dist[x][y] = 487 - np.sqrt((y - img.shape[1]/2)**2 + (x - img.shape[0]/2)**2) 
            mat[x][y] = fitfunc2(dist[x][y], fit_params[0], fit_params[1], fit_params[2], fit_params[3])

    image = np.subtract(img, mat)
    image[image<0] = 0
    minValue, maxValue,_,_ = cv2.minMaxLoc(image)
    image = image * 256/1500
    cv2.imwrite(filepath, image)
    plt.show()
    return image


def removeBackgroundFit2(img, filepath = "./HGE.jpg", offset_range = 20, verbose = True):
    x = np.arange(975)
    x2 = np.concatenate((x[0:440], x[535:975]))
    b = np.zeros((img.shape[0],), dtype=int)
    r_values = []
    alias = ["PrIrO"]
    index = 0
    
    
    for offset in range(offset_range):
        print(offset)
        b = np.zeros((img.shape[0],), dtype=int)
        b[int(img.shape[0] / 2) + offset - int(offset_range/2)]=1
        mod = np.dot(b,img)
        mod2 = np.concatenate((mod[0:440], mod[535:975]))
        fit_params, pcov = scipy.optimize.curve_fit(zweidim, x2, mod2)
        residuals = mod2- zweidim(x2, *fit_params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mod2-np.mean(mod2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(r_squared)
        r_values.append(r_squared)
        
    b = np.zeros((img.shape[0],), dtype=int)
    r_values = np.array(r_values)
    print(r_values.argmax())
    print(int(img.shape[0] / 2) + r_values.argmax() - int(offset_range/2))
    
    b[int(img.shape[0] / 2) + r_values.argmax() - int(offset_range/2)]=1
    mod = np.dot(b,img)
    mod2 = np.concatenate((mod[0:440], mod[535:975]))
    fit_params, pcov = scipy.optimize.curve_fit(zweidim, x2, mod2)
    residuals = mod2 - zweidim(x2, *fit_params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mod2-np.mean(mod2))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)
    
    if verbose:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(x, mod, label=alias[index], s=1, alpha=1)
        
        y_fit = zweidim(x, *fit_params)
        ax.plot(x, y_fit, label=alias[index]+' fit (' + str(round(r_squared,4)) + ')', linewidth="1", color = 'black')
        ax.set(xlabel='Horizontale Position [Pixel]', ylabel='Wert [W.E.]')
        ax.grid()
        ax.set_xlim([0,975])
        plt.legend(loc="upper left")
            
    
    mat = np.zeros(img.shape, dtype=int)
    dist = np.zeros(img.shape, dtype=int)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            dist[x][y] = 487 - np.sqrt((y - img.shape[1]/2)**2 + (x - img.shape[0]/2)**2) 
            mat[x][y] = zweidim(dist[x][y], fit_params[0], fit_params[1], fit_params[2], fit_params[3])

    image = np.subtract(img, mat)
    image[image<0] = 0
    minValue, maxValue,_,_ = cv2.minMaxLoc(image)
    #image = image * 256/1500
    cv2.imwrite(filepath, image)
    plt.show()
    return image





"""

#x2 = np.concatenate((x[0:450], x[535:975]))

filename="HintergrundAbziehen_test"
namelist = ["XRD20210122_20kV_30mA_B1_2000sec_39mm.tif"]#, "13_Simulation/0001.tif"]
alias = ["XRD20210122"]#, "Simulation"]
fitlist = []

index = 0

img = cv2.imread("/home/marvin/Nextcloud/Uni/Semester_7/Bachelorarbeit_Marvin/05_Bilder/Li2Mn2SrN2_LHSJ002A/XRD20210122_20kV_30mA_B1_2000sec_39mm.tif", -1)
b = np.zeros((img.shape[0],), dtype=int)
b[int(img.shape[0] / 2) + 5]=1

mod = np.dot(b,img)
#mod2 = np.concatenate((mod[0:450], mod[535:975]))
mod2 = mod[0:440]
ax.scatter(x, mod, label=alias[index], s=1, alpha=1)
#ax.scatter(x2, mod2, label=name, s=1)


fit_params, pcov = scipy.optimize.curve_fit(fitfunc2, x2, mod2)

residuals = mod2- fitfunc2(x2, *fit_params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((mod2-np.mean(mod2))**2)
r_squared = 1 - (ss_res / ss_tot)

y_fit = fitfunc2(x, *fit_params)
ax.plot(x, y_fit, label=alias[index]+' fit (' + str(round(r_squared,3)) + ')', linewidth="1", color = 'black')
fitlist.append(fit_params)

mat = np.zeros(img.shape, dtype=int)
dist = np.zeros(img.shape, dtype=int)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        dist[x][y] = 487 - np.sqrt((y - img.shape[1]/2)**2 + (x - img.shape[0]/2)**2) 
        mat[x][y] = fitfunc2(dist[x][y], fit_params[0], fit_params[1], fit_params[2], fit_params[3])
#mat = (np.subtract(mat, 321))
image = np.subtract(img, mat)

image[image<0] = 0

minValue, maxValue,_,_ = cv2.minMaxLoc(image)
#print(cv2.minMaxLoc(image))
#globalMin = np.ones(img.shape)
#globalMin = globalMin * minValue
#image = np.subtract(image , globalMin)
   # image = image * int(65535/(maxValue-minValue))

#gray = (image/256).astype('uint8')
image = image * 256/1500

cv2.imwrite("./testimage.png", image)
#fig.colorbar(mat)



ax.set(xlabel='Horizontale Position [Pixel]', ylabel='Wert [W.E.]')
ax.grid()
ax.set_xlim([0,500])
ax.set_ylim([0,1500])
plt.legend(loc="upper left")
print(fitlist)



fig.savefig(filename + ".pdf")
fig2, ax2 = plt.subplots(figsize=(8,8))

plt.imshow(image, cmap='gray')

plt.show()


"""

