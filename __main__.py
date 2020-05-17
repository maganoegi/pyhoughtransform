
# Author: Sergey Platonov
# HEPIA ITI 2020 sem. 4
# Prof: Adrien Lescourt
# Description: Python implementation of the Hough Transform in image processing

# sources used: https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value

import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, sin, cos, radians
import argparse

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

import progress_bar

BLUE = [255,0,0]
GREEN = [0,255,0]
RED = [0,0,255]

def get_Hough_accumulator(img, allow_mapping) -> (np.array, list):
    """ 
        * Constructs the Hough accumulator by going over every angle for every pixel in the image, 
        incrementing ro values in function of theta.  
        * 2 modes: 
            * "mapping" which allows us to skip the parts where no edges were detected.
            * "normal" where we treat every pixel.
        * saves the ro values for every pixel, to avoid recalculating after.
    """    

    height, width = img.shape
    ro_max = int(sqrt(width * height))

    # init the accumulator
    accumulator = np.zeros((NB_ANGLES, ro_max * 2))

    # contains the ro values that we find. avoids recalculation later on, when we draw the curves
    ros = []

    # prepare the triginometric values ahead of time, to avoid them being recalculated every iteration
    cosines = [cos(radians(angle)) for angle in ANGLES]
    sines = [sin(radians(angle)) for angle in ANGLES]

    # go over every pixel value
    for x in range(height):
        ro_rows = []
        for y in range(width):
            ro_pixels = []
            # go over every possible angle theta for the pixel
            if not (allow_mapping and img[x][y] == 0):
                for i in range(NB_ANGLES):
                    ro = int((x * cosines[i]) + (y * sines[i]))
                    if img[x][y] != 0:
                        accumulator[i][ro] += 1

                    ro_pixels.append(ro)
                            
            ro_rows.append(ro_pixels)
        ros.append(ro_rows)

        percentage = (x * 100) // (height * 2)
        if allow_pb: progress_bar.draw_progress_bar(percentage, "build acc")
    
    return accumulator, ros


def draw_lines(edges, maxima, original, allow_mapping, ros) -> None:
    """ 
        * Uses the Hough accumulator to project the lines onto the source image. 
        * 2 modes: 
            * "mapping" which allows us to skip the parts where no edges were detected.
            * "normal" where we treat every pixel.
        * utilises the ro values calculated alongside the accumulator.
    """

    height, width = edges.shape

    for x in range(height):
        for y in range(width):
            if allow_mapping and edges[x][y] == 0:
                # when we are only interested in edges that contain an edge
                continue

            for i in range(NB_ANGLES):
                ro = ros[x][y][i]

                if maxima[i][ro]: 
                    original[x][y] = GREEN

        percentage = 50 + (x * 100) // (2 * height)
        if allow_pb: progress_bar.draw_progress_bar(percentage, "draw lines")


def get_maxima(accumulator) -> np.array:
    """ 
        * gets the maxima in a 2d np array. 
        * returns: np.array of bools, representing our maxima and their coordinates
    """
    n = int(len(accumulator) * 0.15) # dynamically fixes the dimensions of the filter

    # apply the filters and do matrix operations
    data_max = filters.maximum_filter(accumulator, n)
    maxima = (accumulator == data_max)
    data_min = filters.minimum_filter(accumulator, n)
    diff = ((data_max - data_min) > hough_thresh * accumulator.max())
    maxima[diff == 0] = 0

    # draw the maxima onto the Hough space representation of the accumulator
    labeled, _ = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    plt.axis('off')
    plt.imshow(accumulator, cmap='gray')
    plt.autoscale(False)
    plt.plot(x, y, 'r.')
    plt.savefig('maxima.png', bbox_inches = 'tight')

    return maxima

def parse_input() -> (str, int, int, float, float, bool, bool):
    """ 
        argument parser wrapper. Terminal commands and parameters are defined here.
        * useful for this implementation, as it allows for easy parameter tuning for learning purposes
    """
    parser = argparse.ArgumentParser(description="Apply the Hough Transform on the target image")
    parser.add_argument('filename', type=str, help='filename to be opened')
    parser.add_argument('-cmin', '--canny_min', type=int, default=100, help='minimal threshold value for the Canny edge detection')
    parser.add_argument('-cmax', '--canny_max', type=int, default=255, help='maximal threshold value for the Canny edge detection')
    parser.add_argument('-hthresh', '--hough_thresh', type=float, default=0.35, help='percentage value used for Hough transform sensitivity')
    parser.add_argument('-s', '--step', type=float, default=0.5, help='helps tune the precision and speed of the algorithm. Increase the value to increase performance (default 0.5)')
    parser.add_argument('-m', '--map', default=False, action='store_true', help='maps the lines to the actual image - unlike the hough transform that draws the entire lines')
    parser.add_argument('-b', '--bar', default=False, action='store_true', help='allows the progress bar (may require dependencies!)')
    args = parser.parse_args()

    return args.filename, args.canny_min, args.canny_max, args.hough_thresh, args.map, args.step, args.bar


if __name__ == "__main__":

    filename, canny_min, canny_max, hough_thresh, allow_mapping, angle_step, allow_pb = parse_input()

    MAX_ANGLE = 360.0
    NB_ANGLES = int(angle_step * MAX_ANGLE)
    ANGLES = np.linspace(0.0, MAX_ANGLE, NB_ANGLES)


    print(f"Canny Min: {canny_min}\nCanny Max: {canny_max}\nHough threshold: {hough_thresh}\nNb of angles: {NB_ANGLES}\nMap only on edges? {allow_mapping}")

    print("\nimage preprocessing...")
    img = cv2.imread(filename)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("\nImage with edges saved...")
    edges = cv2.Canny(img, canny_min, canny_max)
    plt.imsave("edges.png", edges, cmap='gray')

    if allow_pb: progress_bar.init(color=True, dynamic=False, spinner=0, empty=True) # use the -p flag to enjoy the progress bar that guides you through this long process..

    print("\ncalculating the Hough accumulator...")
    accumulator, ros = get_Hough_accumulator(edges, allow_mapping)
    cv2.imwrite("accumulator.png", accumulator)

    print("\nGetting the local maxima...")
    maxima = get_maxima(accumulator)

    print("\nDrawing the lines onto the original image...")
    draw_lines(edges, maxima, img, allow_mapping, ros)

    print("\nsaving result image...")
    cv2.imwrite("final.png", img)

    print("\n...done\nrelevant images saved as: edges.png accumulator.png maxima.png final.png")
    if allow_pb: progress_bar.destroy()
