import cv2

def view_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)


# img = cv2.imread("./res/poorly_lit_1.png")
img = cv2.imread("./res/poorly_lit_2.jpg")
#   using fx and fy maintains the aspect ratio
factor = 0.5

#   img.shape returns 3 valus when not gray scaled, but 2 when gray scaled
# w, h = img.shape[-2::-1]    #   not grayscaled

img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
view_image(img, "Original Image")


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# w, h = img.shape[::-1]       #    grayscaled


#   apply binary thresholding
"""
    cv2.THRESH_BINARY
    cv2.THRESH_BINARY_INV
    cv2.THRESH_TRUNC
"""
_, result = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)


"""
    Adaptive Thresholding

    A more robust way of applying thresholding on pixels by taking into account the 
    surrounding pixels of the current pixel

    arg1: image
    arg2: max value
    arg3: method
        Different Methods
        *   mean adaptive method:   the threshold produced takes into account the mean of the current pixel's surrounding
        neighbours --- cv2.ADAPTIVE_THRESHOLD_MEAN_C
        *   gaussian method: uses gaussian weighting using current pixel's surrounding neighbours
            to determine the threshold of the current pixels
            --- cv2.ADAPTIVE_THRESHOLD_GAUSSIAN_C
    arg4: threshold type: e.g. cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC
    arg5: block size -- number of surrounding neighbours to consider -- MUST be ODd number
    arg6: constant to reduce noise ---- the higher it is, the more noise is removed
"""
#   Parameters usef in tutorial
# adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                               cv2.THRESH_BINARY, 41, 5)

adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 41, 5)

view_image(result, "Improved Readability using Normal Thresholding")
view_image(adaptive_result, "Improved Readability using Adaptive Thresholding")

cv2.destroyAllWindows()