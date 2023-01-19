# Import numpy and OpenCV
import numpy as np
import cv2
import glob
import os
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def frame_to_vid(frame_path,output_vid, vn = 'converted'):
    '''
    frame_path {str}: path to folder which has frames extracted
    vn {str}: Video name 

    '''
    img_array = []
    csvs = glob.glob(frame_path+ '/*.png')

    sort_nicely(csvs)

    for filename in csvs:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    outer = cv2.VideoWriter(output_vid+'/'+ vn + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
    
    for i in range(len(img_array)):
        outer.write(img_array[i])
    outer.release()
