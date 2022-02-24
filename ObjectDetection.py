import cv2 as cv
import numpy as np

# loads images
haystack_img = cv.imread('starcraftBase.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('commandCenter.jpg', cv.IMREAD_UNCHANGED)

# finds imagine in haystack
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# best match positions 
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
print('Best match top left position: ', str(max_loc))
print('Best match confidence: ', max_val)

threshold = 0.8
if max_val >= threshold:
    print('Needle found')

    # get the dimensions of the image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    # draws box around object
    cv.rectangle(haystack_img, top_left, bottom_right, color = (0,255,0), thickness = 2, lineType = cv.LINE_4)
    cv.imshow('Result', haystack_img)
    cv.waitKey()
else:
    print('Needle not found')
