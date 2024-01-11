import time

from algorithms import nearest_neighbour as nninterp
from algorithms import bilinear_interpolation as blinterp
from algorithms import bicubic_interpolation as bcinterp
import cv2

img = cv2.imread("data/squirtle.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

start = int(time.time())

newImg = nninterp.perform2x(img)
cv2.imwrite("result/up_nni2x.png", newImg)
print("nninterp2x done in:", -start + (start := time.time()), "seconds")
newImg = nninterp.perform4x(img)
cv2.imwrite("result/up_nni4x.png", newImg)
print("nninterp4x done in:", -start + (start := time.time()), "seconds")
newImg = blinterp.perform2x(img)
cv2.imwrite("result/up_bli2x.png", newImg)
print("blinterp2x done in:", -start + (start := time.time()), "seconds")
newImg = blinterp.perform4x(img)
cv2.imwrite("result/up_bli4x.png", newImg)
print("blinterp4x done in:", -start + (start := time.time()), "seconds")
newImg = bcinterp.perform2x(img)
cv2.imwrite("result/up_bci2x.png", newImg)
print("bcinterp2x done in:", -start + (start := time.time()), "seconds")
