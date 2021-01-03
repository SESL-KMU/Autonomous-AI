import cv2  # opencv 사용

image = cv2.imread('road.png')  # 이미지 읽기
height, width = image.shape[:2]  # 이미지 높이, 너비

kernel_size = 3
low_threshold = 70
high_threshold = 210
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 흑백이미지로 변환
blur_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)  # Blur 효과
canny_img = cv2.Canny(blur_img, low_threshold, high_threshold)  # Canny edge 알고리즘

cv2.imshow('origin', image)
cv2.imshow('result', canny_img)  # Canny 이미지 출력
cv2.waitKey(0)