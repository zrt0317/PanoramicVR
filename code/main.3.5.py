import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.interpolate


start_time = time.time()


filename1 = '2_1.jpg'
filename2 = '2_2.jpg'


KEYPOINTS_CALC_MATCH = 1
if KEYPOINTS_CALC_MATCH:
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
    matchImg = cv2.drawMatches(p1, kp1, p2, kp2, matches, p1, 
            matchColor=(0,0,0), singlePointColor=(0,0,0), flags=2)
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
    

MOULDS_CALC = 1
if MOULDS_CALC:
    ''' CALCULATE THE PARAMETERS OF MOULDS '''
    _, aay = np.median(dif, axis=0)  # x-axis and y-axis bias.
    sy = p1.shape[0]
    aay = int(aay%sy+sy)
    p11 = np.vstack([p1, p1])
    p0 = np.vstack([p2, p2, p2, p2])
    uy = sy//8


KEYPOINTS_CALC_MOULD = 1
if KEYPOINTS_CALC_MOULD:
    ''' WITH MOULD: KEYPOINTS CALCULATION '''
    mtPx = np.array([1000 for i in range(15)]).reshape(3,5)
    
    for i in range(8):
        for j in range(3):
            xstart = j*192
            xend = xstart+256
            
            tpl_node = i*uy
            tpl = p11[tpl_node:int(tpl_node+uy*1+70), xstart:xend, :]
            tgt_node = i*uy+aay-int(uy*0.01+70)
            tgt = p0[tgt_node:int(tgt_node+uy*1.02+140), xstart:xend, :]
            
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(cv2.cvtColor(
                    tpl, cv2.IMREAD_ANYCOLOR), None)
            kp2, des2 = orb.detectAndCompute(cv2.cvtColor(
                    tgt, cv2.IMREAD_ANYCOLOR), None)
        
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            if des1 is not None and des2 is not None:
                matches1 = bf.match(des1,des2)
                matches1 = [i for i in matches1 if i.distance<48]
                matches1 = sorted(matches1, key = lambda x:x.distance)
                
                # Initialize lists
                mylst = []
                # For each match...
                for mat in matches1:
                    # Get the matching keypoints for each of the images
                    img1_idx = mat.queryIdx
                    img2_idx = mat.trainIdx
                    # x - columns; y - rows; Get the coordinates
                    (x1,y1) = kp1[img1_idx].pt
                    (x2,y2) = kp2[img2_idx].pt
                    # Append to list
                    mylst.append([x1+xstart,
                                  x2+xstart+sy//2,
                                  mat.distance,
                                  (y1+tpl_node)%sy,
                                  (y2+tgt_node)%sy])
                
                mylst = np.round(mylst, 0)
                
                if mylst.any():
                    mtPx = np.concatenate((mtPx, mylst), axis=0)


    mtPx = mtPx[mtPx[:, 2].argsort()]   #按照第3列 '特征点的差距' 对行排序


print(time.time()-start_time)


MATCH_RESULT_PIC = 1
if MATCH_RESULT_PIC:
    # 绘制匹配结果
    p_out = np.hstack([p1,p2])[:, :, 2::-1]
    plt.figure(figsize=(11,11))
    plt.axis('off')
    plt.imshow(p_out)
    for i in range(mtPx.shape[0]):
        plt.plot(mtPx[i][:2], mtPx[i][-2:], color='k', lw=1)


KEYPOINT_CLUSTER = 1
if KEYPOINT_CLUSTER:
    X = np.concatenate((
            np.cos(mtPx[:,[0,3]]*np.pi/p1.shape[1]),
            np.sin(mtPx[:,3:4]*np.pi/p1.shape[1])), axis=1)
        
    nclusters = 10
    estimator = KMeans(n_clusters=nclusters)  # 构造聚类器
    estimator.fit(X)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
        
    P1pts = mtPx[:, [0,3]]
    P2pts = mtPx[:, [1,4]]-np.array([p1.shape[1],0])
    p1pts = []
    p2pts = []
    for i in range(nclusters):
        p1pts.append(np.median(P1pts[label_pred == i,:], axis=0))
        p2pts.append(np.median(P2pts[label_pred == i,:], axis=0))
        
    out = np.concatenate((np.round(p1pts, 0), np.round(p2pts, 0)), axis=1)
    out0 = out[out[:,1].argsort()]
    out1 = np.concatenate((
            out0[-3:,:]-[0,p1.shape[0],0,p1.shape[0]],
            out0,
            out0[:3,:]+[0,p1.shape[0],0,p1.shape[0]]
            ), axis=0)


COLLABORATION_PIC = 1
if COLLABORATION_PIC:
    # 绘制匹配结果需要用的背景图
    p_out = np.hstack([p1,p2])[:, :, 2::-1]
    plt.figure(figsize=(11,11))
    plt.axis('off')
    plt.imshow(p_out)
    
    # 三次样条
    x = out1[:, 1]
    y = np.sort(out1[:, 3], kind='quicksort', order=None)
    f = scipy.interpolate.interp1d(x, y, kind=3)
    
    slist = [440, 430, 380, 420, 350, 400, 370, 330, 400, 450]
    tlist = [150, 280, 390, 430, 600, 730, 910, 960, 1040, 1130]
    for kk in range(10):
        s = slist[kk]
        t = tlist[kk]
        
        for i in range(1, out1.shape[0]):
            if t<out1[i, 1]:
                break
    
        s_out = int(s+(-out1[i-1, 0]+out1[i-1, 2]-out1[i, 0]+out1[i, 2])/2)
        t_out = int(f(t))%p1.shape[0]
        
        # 圆形
        r = 0.02*p1.shape[0]
        theta = np.arange(0, 2*np.pi, 0.01)
        xcircle = s_out + r * np.cos(theta) + p1.shape[1]
        ycircle = t_out + r * np.sin(theta)
        
        # 绘制匹配结果
        plt.plot([s, s_out+p1.shape[1]],
                 [t, t_out%p1.shape[0]],
                 color='r', lw=1)
        plt.plot(xcircle,
                 ycircle%p1.shape[0],
                 color='r', lw=1)
        ''