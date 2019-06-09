#模板匹配
import cv2


def template_demo():
    tpl = cv2.imread("P3a.jpg")
    target = cv2.imread("P1b.png")
    
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    #3种模板匹配方法
    
    th, tw = tpl.shape[:2]
    for md in methods:
        result = cv2.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if md == cv2.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)   #br是矩形右下角的点的坐标
        cv2.rectangle(target, tl, br,
                      ((md==1)*255, (md==3)*255, (md==5)*255), 1)
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", target)
    cv2.imwrite('1.jpg', target)


template_demo()
cv2.waitKey(0)
cv2.destroyAllWindows()
