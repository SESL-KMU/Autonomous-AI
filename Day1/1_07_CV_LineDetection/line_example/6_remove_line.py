import cv2  # opencv
import numpy as np
from google.colab.patches import cv2_imshow


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    return lines


def weighted_img(img, initial_img, aa=1, bb=1., cc=0.):
    return cv2.addWeighted(initial_img, aa, img, bb, cc)


image = cv2.imread('road.png')
height, width = image.shape[:2]

kernel_size = 3
low_threshold = 70
high_threshold = 210
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
canny_img = cv2.Canny(blur_img, low_threshold, high_threshold)

vertices = np.array(
    [[(0, height), (width / 2 - 70, height / 2 + 0), (width / 2 + 70, height / 2 + 0), (width - 0, height)]],
    dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices)

line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)
line_arr = np.squeeze(line_arr)

slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

line_arr = line_arr[np.abs(slope_degree) < 160]
slope_degree = slope_degree[np.abs(slope_degree) < 160]
line_arr = line_arr[np.abs(slope_degree) > 95]
slope_degree = slope_degree[np.abs(slope_degree) > 95]
L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
L_lines, R_lines = L_lines[:, None], R_lines[:, None]
draw_lines(temp, L_lines)
draw_lines(temp, R_lines)

result = weighted_img(temp, image)
cv2_imshow(result)
cv2.waitKey(0)