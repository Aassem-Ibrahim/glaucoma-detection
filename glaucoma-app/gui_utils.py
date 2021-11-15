from PyQt5.QtGui import QImage as qi
import numpy as np
import cv2


# NUMPY COLOR ARRAYS
WHITE = np.array([255, 255, 255, 255])
LIGHT_GRAY = np.array([129, 129, 129, 255])
DARK_GRAY = np.array([3, 3, 3, 255])
BLACK = np.array([0, 0, 0, 255])
TRANSPARENT = np.array([255, 255, 255, 0])


def get_color_channels(rgb):
    ''' Get color channels of a color as a string '''
    # Remove the '#' symbol
    rgb = rgb[1:]
    # Split color channels
    return tuple(int(rgb[i: i+2], 16) for i in range(0, 6, 2))


def grayscale_color(rgb):
    ''' Calculate grayscale color of an RGB color (string to string) '''
    # Remove the '#' symbol
    rgb = rgb[1:]
    # Calculate average of RGB channels
    average = sum([int(rgb[i: i+2], 16) for i in range(0, 6, 2)]) // 3
    # Convert average to hex
    grayscale = f'{average:02x}'
    # Return grayscale color
    return '#' + grayscale * 3


def get_boundary_info(img, c1, c2):
    ''' Get boundary of specific colors '''
    # Image threshold
    ret, thresh = cv2.threshold(img, c1, c2, cv2.THRESH_BINARY)
    # Get contours
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    # Return last rectange in contours (as the first is the image boundry)
    #   also return its area
    return cv2.boundingRect(contours[-1]), cv2.contourArea(contours[-1])


def get_boundaries_info(img):
    ''' Get boundaries of of mask '''
    # Convert image to grayscale
    pix = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # Get boundries for ground truth image
    return (get_boundary_info(pix, 127, 255),
            get_boundary_info(pix, 128, 255))


def colorize_mask(cv2i, scale, disc_alpha, cup_alpha, disc_color, cup_color):
    ''' Colorize mask using color and alpha and scale if passed '''
    # Get color channels of each mask layer
    dsc_r, dsc_g, dsc_b = get_color_channels(disc_color)
    cup_r, cup_g, cup_b = get_color_channels(cup_color)
    # Add alpha channel to image (convert)
    pix = cv2.cvtColor(cv2i, cv2.COLOR_BGR2BGRA)
    # Scale image if scale passed
    if scale is not None:
        pix = cv2.resize(pix, (scale, scale))
    # Get masks of each region (background, disc, cup)
    mask_bgd = cv2.inRange(pix, LIGHT_GRAY, WHITE)
    mask_dsc = cv2.inRange(pix, DARK_GRAY, LIGHT_GRAY)
    mask_cup = cv2.inRange(pix, BLACK, DARK_GRAY)
    # Colorize masks
    pix[mask_bgd > 0] = TRANSPARENT
    pix[mask_dsc > 0] = np.array([dsc_b, dsc_g, dsc_r, disc_alpha])
    pix[mask_cup > 0] = np.array([cup_b, cup_g, cup_r, cup_alpha])
    # Return edited image as QImage
    return cvImage_to_qImageA(pix)


def edit_contrast(cv2i, value=0):
    ''' Edit contrast of CV2 image '''
    # Map value between [-127, 127] instead of [-255, 255]
    value //= 2
    # Make sure value is not zero (zero = no change)
    if value != 0:
        # FROM WIKIPEDIA
        f = 131 * (value + 127) / (127 * (131 - value))
        cv2i = cv2.addWeighted(cv2i, f, cv2i, 0, 127 * (1 - f))
    # Return image after edit
    return cv2i


def edit_image_of(cv2i, key, value=0):
    ''' Edit HSV channels '''
    # Make sure value is not zero (zero = no change)
    if value != 0 or key == 2:
        # Convert RGB image to HSV
        hsv = cv2.cvtColor(cv2i, cv2.COLOR_BGR2HSV)
        # Edit HSV image
        hsv[:, :, key] = cv2.add(hsv[:, :, key], value)
        # Convert image back to RGB and return
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return cv2i


def cvImage_to_qImage(cv2i):
    ''' Convert CV2Image to QImage '''
    # Get channels
    height, width, channel = cv2i.shape
    # Convert to QImage
    return qi(cv2i.data,
              width,
              height,
              channel * width,
              qi.Format_RGB888).rgbSwapped()


def cvImage_to_qImageA(cv2i):
    ''' Convert CV2Image to QImage with alpha channel '''
    # Convert RGB to RGBA (with alpha)
    cv2i = cv2.cvtColor(cv2i, cv2.COLOR_RGB2RGBA)
    # Get channels
    height, width, channel = cv2i.shape
    # Convert to QImage
    return qi(cv2i.data,
              width,
              height,
              channel * width,
              qi.Format_RGBA8888).rgbSwapped()


def qImage_to_cvImage(qpix):
    ''' Convert QImage to CV2Image '''
    # Get width and height of image
    h, w = qpix.size().width(), qpix.size().height()
    # Get QImage from QPixmap
    qimg = qpix.toImage()
    # Convert QImage to byteStr
    byte_str = qimg.bits().asstring(w * h * 4)
    # Create Numpy array from byteStr buffer
    return np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))


def blur_qImage(qpix, ksize):
    ''' Blur QImage '''
    # Convert QImage to CV2Image without alpha channel
    cv2i = qImage_to_cvImage(qpix)[:, :, :3]
    # Add blur to CV2Image, then convert back to QImage
    return cvImage_to_qImage(cv2.blur(cv2i, ksize))


if __name__ == '__main__':
    EDIT_HUE, EDIT_SAT, EDIT_BRI = 0, 1, 2

    image_filename = 'glaucoma-cases/V0001.jpg'

    from time import time
    t0 = time()

    img = cv2.imread(image_filename)
    t1 = time()
    print(f'It took {t1-t0:.5f} to load the CV2 image.')

    new = cvImage_to_qImage(img)
    t2 = time()
    print(f'It took {t2-t1:.5f} to convert to qImage.')

    print(f'\n{"-"*30}\n')
    out1 = edit_contrast(img, 127)
    t3 = time()
    print(f'It took {t3-t2:.5f} to edit contrast.')

    out2 = edit_image_of(img, EDIT_HUE, 127)
    t4 = time()
    print(f'It took {t4-t3:.5f} to edit hue.')

    out3 = edit_image_of(img, EDIT_SAT, 127)
    t5 = time()
    print(f'It took {t5-t4:.5f} to edit saturation.')

    out4 = edit_image_of(img, EDIT_BRI, 127)
    t6 = time()
    print(f'It took {t6-t5:.5f} to edit brightness.')

    input('\nPress <Enter> to exit...')
