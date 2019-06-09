from numpy import zeros
from skimage import io
from copy import deepcopy
from math import atan,sqrt,pi,floor
import os


def coordinateToAngle(x,y,z):
    fi = atan(z/sqrt(x*x+y*y))
    if y==0:
        theta = pi*((x==-1)+1/2)
    else:
        theta = atan(-x/y)+pi*(y<0)+2*pi*(x>0)*(y>0)
    return [theta, fi]


''' read picture and set parameters '''
filename = 'x.jpg'
img = io.imread(filename)
aaa = deepcopy(img)*0     # used for marking the positions of panorama
#io.imshow(img)
n = 301         # size of small pictures
s = 0.55        # sine parameter suggested between .4 and .6
c = sqrt(1-s**2)
a = zeros((n,n,2))    # used to save location info.

curpath = os.getcwd()
folderpath = curpath+'\\'+filename.split('.')[0]
if not os.path.exists(folderpath):
    os.mkdir(folderpath)
print(-1,": basic setting finished.")
    
    
''' calculate the distortion (northern) '''
for i in range(n):
    for j in range(n):
        x = (0.5-j/(n-1))
        y = +sqrt(3/4)*s+(i*c/(n-1)-c/2)
        z = +sqrt(3/4)*c-(i*s/(n-1)-s/2)
        a[i][j] = coordinateToAngle(x,y,z)
print(-1,": distortion calculation finished.")


''' voluation (northern) '''
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
ssps = 6    # pctures in single side

for flag in range(ssps):
    b = zeros((n,n,3), dtype='uint8')
    ax = fig.add_subplot(231+flag)
    ax.axis('off')
    
    for i in range(n):
        for j in range(n):
            row = floor(1344-a[i][j][1]*2688/pi)
            col = floor(a[i][j][0]*2688/pi+flag*5376//ssps)%5376
            b[i][j] = img[row][col]
            if flag%2+1:
                aaa[row][col]=[i%100+155,j%100+155,(flag%2)*100+155]
    
    ax.imshow(b)
    io.imsave(folderpath+'\\N'+str(flag)+'.jpg',b)
    print(flag)

plt.show()


''' calculate the distortion (southern) '''
a = [[[a[n-1-i][j][0], -a[n-1-i][j][1]]for j in range(n)]for i in range(n)]


''' voluation (southern) '''
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
ssps = 6    # pctures in single side

for flag in range(ssps):
    b = zeros((n,n,3), dtype='uint8')
    ax = fig.add_subplot(231+flag)
    ax.axis('off')
    
    for i in range(n):
        for j in range(n):
            row = floor(1344-a[i][j][1]*2688/pi)
            col = floor(a[i][j][0]*2688/pi+flag*5376//ssps)%5376
            b[i][j] = img[row][col]
            if flag%2+1:
                aaa[row][col]=[i%100+155,j%100+155,(flag%2)*100+155]
    
    ax.imshow(b)
    io.imsave(folderpath+'\\S'+str(flag)+'.jpg',b)
    print(flag)

plt.show()


#plt.imshow(aaa)
#io.imsave(folderpath+'\\ttn.jpg',aaa)
