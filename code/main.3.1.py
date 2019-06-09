import cv2
import time


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
