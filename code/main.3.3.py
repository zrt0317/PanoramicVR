import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


a=time.time()


filename1 = '2_1.jpg'
filename2 = '2_2.jpg'


if 1:
    ''' KEYPOINTS CALCULATION & MATCHING '''
    p1 = cv2.imread(filename1)
    p2 = cv2.imread(filename2)
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(
            p1, cv2.IMREAD_GRAYSCALE), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(
            p2, cv2.IMREAD_GRAYSCALE), None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance) 
    
    '''
    matchImg = cv2.drawMatches(p1, kp1, p2, kp2, matches, p1, flags=2)
    cv2.imwrite(str(int(time.time()))+'.jpg', matchImg)
    '''
    
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
    

if 1:
    ''' CALCULATE THE PARAMETERS OF MOULDS '''
    _, aay = np.median(dif, axis=0)  # x-axis and y-axis bias.
    sy = p1.shape[0]
    aay = int(aay%sy+sy)
    p0 = np.vstack([p2, p2, p2, p2])
    uy = sy//4
    
    
if 1:
    ''' WITH MOULD: KEYPOINTS CALCULATION '''
    for i in range(4):
        tpl_node = i*uy
        tpl = p1[tpl_node:int(tpl_node+uy*1), :, :]
        tgt_node = i*uy+aay-int(uy*0.01+10)
        tgt = p0[tgt_node:int(tgt_node+uy*1.02+20), :, :]
        
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(
                tpl, cv2.IMREAD_ANYCOLOR), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(
                tgt, cv2.IMREAD_ANYCOLOR), None)
    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches1 = bf.match(des1,des2)
        matches1 = sorted(matches1, key = lambda x:x.distance)
        
        '''
        matchImg = cv2.drawMatches(tpl, kp1, tgt, kp2, matches1, tpl,
                    matchColor=(0,0,0), singlePointColor=(0,0,0), flags=2)
        cv2.imwrite(str(int(time.time()*1e6))+'.jpg', matchImg)
        '''
        
        
        # Initialize lists
        mylst, dif = [], []
        # For each match...
        for mat in matches1:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            # x - columns; y - rows; Get the coordinates
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
            # Append to list
            mylst.append([x1,
                          p1.shape[1]+x2,
                          mat.distance,
                          y1+tpl_node,
                          (y2+tgt_node)%p1.shape[0]])
        
        mylst = np.round(mylst, 0)
        
        if i==0:
            mtPx = mylst
        else:
            mtPx = np.concatenate((mtPx, mylst), axis=0)
            
            
    mtPx1 = mtPx[mtPx[:, 2]<=56, :]         #筛选数组，保留第3列满足条件的行
    mtPx1 = mtPx1[mtPx1[:, 2].argsort()]    #按照第3列 '特征点的差距' 对行排序

    
    # 绘制匹配结果
    p_out = np.hstack([p1,p2])[:, :, 2::-1]
    plt.figure(figsize=(11,11))
    plt.imshow(p_out)
    for i in range(mtPx1.shape[0]):
        plt.plot(mtPx1[i][:2], mtPx1[i][-2:], color='k', lw=1)
    
    
print(-a+time.time())
