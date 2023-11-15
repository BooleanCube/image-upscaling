from algorithms import nninterp
import cv2

img = cv2.imread("/home/boole/Pictures/pfp/finnspace.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

print("started")

newImg = nninterp.perform2x(img)
cv2.imwrite("result/upscaled.png", newImg)

print(img.shape)
print(newImg.shape)
