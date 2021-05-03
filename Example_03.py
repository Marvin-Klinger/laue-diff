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
home the sample holder, then shut down the motor unit.
"""

home()
client_program("S;;;")