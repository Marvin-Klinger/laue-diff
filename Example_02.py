#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marvin Klinger

For additional resources see https://github.com/Marvin-Klinger/laue-diff
"""

"""
######[Imports]################################################################

Everything is imported although it might not be needed.

"""

import cv2
import numpy as np
from Basicfunctions import bracketing, simpleRaster_v2, home, laueClient, maxFinder, longExposure, brightnessAnalysis, client_program
from Rotation import simpleRotation

"""
Important:
    Before first use home the machine with this command. Then adjust the
    position of the sample manually, so it sits approximately in front of the
    aperture.
"""
#home()


"""
Take 25 pictures, with 10s exposure each. The distance between the samples
will be 0.5mm.

After the run the sample will be centered so a long exposure can be taken.
"""

simpleRaster_v2(5,5,0.5,10000,(0,0))