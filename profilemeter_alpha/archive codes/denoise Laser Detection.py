import  cv2
import numpy as np
from  sklearn import  cluster
import time



img = cv2.imread('noise.png')


def laserDetection( img, thresh=120, eps=10, min_samples=5 ):
    b,g,r = cv2.split( img )
    msk = r - np.maximum(b,g).astype( np.int16 )
    msk = np.clip(msk, 0, 255 ).astype(np.uint8 )
    _, msk = cv2.threshold( msk , thresh , 255  , cv2.THRESH_BINARY )
    pts = np.transpose( np.nonzero(msk) )
    clst = cluster.DBSCAN( eps=eps, min_samples = min_samples ).fit(pts )
    lbl = clst.labels_
    lbl[lbl<0] = lbl.max()+1
    hist = np.bincount( lbl )
    biggest_class = np.argmax( hist )
    res_pts = pts[lbl == biggest_class ]
    res_msk = np.zeros_like(msk)
    res_msk[ res_pts[:,0], res_pts[:,1]] = 255
    return res_msk


t = time.time()
res = laserDetection(img)
print( time.time() - t )
cv2.circle(img, (10,250), 5, (255,0,0), thickness=-1)
cv2.imshow('img', img )
cv2.imshow('res', res )

