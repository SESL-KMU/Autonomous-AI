import cv2 # opencv 사용
from google.colab.patches import cv2_imshow

image = cv2.imread('road.png', cv2.IMREAD_GRAYSCALE) # 이미지 읽기

_, thres = cv2.threshold(image,125,255,cv2.THRESH_BINARY)
_, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
adap = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)



print('thres')
cv2_imshow(thres) # 이미지 출력
print('otsu')
cv2_imshow(otsu) # 이미지 출력
print('adap')
cv2_imshow(adap) # 이미지 출력
print('image')
cv2_imshow(image) # 이미지 출력
cv2.waitKey(0)