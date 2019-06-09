import cv2
import time
import numpy as np


filename1 = 'P1b.png'
filename2 = 'P1c.png'


p1 = cv2.imread(filename1)
p2 = cv2.imread(filename2)
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(cv2.cvtColor(
        p1, cv2.IMREAD_GRAYSCALE), None)
kp2, des2 = orb.detectAndCompute(cv2.cvtColor(
        p2, cv2.IMREAD_GRAYSCALE), None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = [i for i in matches if i.distance<48]
matches = sorted(matches, key = lambda x:x.distance)

matchImg = cv2.drawMatches(p1, kp1, p2, kp2, matches[:52], p1, 
        matchColor=(255,0,0), singlePointColor=(255,0,0), flags=2)
cv2.imwrite(str(int(time.time()))+'.jpg', matchImg)


# Initialize lists
mylst, dif = [], []
# For each match...
for mat in matches:
    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx
    # x - columns; y - rows; Get the coordinates
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    # Append to list
    mylst.append([x1, y1, x2, y2])
    dif.append([x1-x2, y1-y2])

mylst = np.round(mylst, 0)
dif = np.round(dif, 0)
