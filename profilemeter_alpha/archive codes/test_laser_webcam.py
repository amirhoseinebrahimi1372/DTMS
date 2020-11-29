import cv2
import numpy as np
import numba
from numba import cuda
import time

UP_LEFT = 0
UP_RIGHT = 1
DOWN_LEFT = 2
DOWN_RIGHT = 3
RED = 5
BLUE = 6
KEYS = [ 'UL','UR', 'DL', 'DR']

#_______________________________________________________________________________________________________
#                                               Laser Detection
#_______________________________________________________________________________________________________
@cuda.jit('void(uint8[:,:,:], uint8,uint8, uint8[:,:])')
def laser_detection_cuda( img, thresh,  color, result ):

    i, j = cuda.grid(2)
    img_rows, img_cols = img.shape[:2]
    
    if (i >= img_rows) or (j >= img_cols):
        return


    '''
    if color == 'blue' or color == 'b':
        for key in range(4) :
            max_gr = 50#np.maximum( imgs[key][i,j,1], imgs[key][i,j,0])
            res = imgs[key][i,j,0] - max_gr
            if res >= thresh:
                res = 255
            else :
                res = 0
                
            results[key][i,j]= res


    elif color == 'red' or color == 'r':
    '''
    if color == RED:
        res = img[i, j, 2] - (img[i,j,0] +  img[i,j,1])/2
        if res >= thresh:
            res = 255
        else :
            res = 0
            
        result[i,j]= res


    elif color == BLUE:
        res = img[i, j, 0] - (img[i,j,1] +  img[i,j,2])/2
        if res >= thresh:
            res = 255
        else :
            res = 0
            
        result[i,j]= res


        
#@numba.jit
def laser_detection( img, thresh,  color ):

    img_cuda = cuda.to_device(img)
    reslut = np.zeros( img.shape[:2], img.dtype )
    result_cuda = cuda.to_device(reslut)
    
    laser_detection_cuda[griddim, blockdim]( img_cuda, thresh, color, result_cuda)
    '''
    if color == BLUE:
        res = img[:,:,0] - ( img[:,:,1] + img[:,:,2] ) / 2
        res[res<0] = 0
        res = res.astype( np.uint8 )
        _, res = cv2.threshold( res, thresh, 255, cv2.THRESH_BINARY)
        return res

    if color == RED:
        res = img[:,:,2] - ( img[:,:,0] + img[:,:,1] ) / 2
        res[res<0] = 0
        res = res.astype( np.uint8 )
        _, res = cv2.threshold( res, thresh, 255, cv2.THRESH_BINARY)
        return res
    '''
    return result_cuda.copy_to_host()
    

#_______________________________________________________________________________________________________
#
#_______________________________________________________________________________________________________





choose = RED
thresh_red = 50
thresh_blue = 50

if __name__ == '__main__':
    
    img = cv2.imread('color_test.png')

    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        cv2.imshow('frame', frame )

        blockdim = (32, 32)
        griddim = (frame.shape[0] // blockdim[0] + 1, frame.shape[1] // blockdim[1] + 1)


        red_laser = laser_detection(frame, thresh_red, RED)
        blue_laser = laser_detection(frame, thresh_blue, BLUE)

        cv2.imshow( 'RED', red_laser )
        cv2.imshow( 'BLUE', blue_laser )
        
        key  = cv2.waitKey(30)

        if key == ord('r'):
            print('red choosen')
            choose = RED

        elif key == ord('b'):
            print('blue choosen')
            choose = BLUE

        elif key == 45 : #-:
            if choose == BLUE:
                thresh_blue -= 2
            elif choose == RED:
                thresh_red -= 2


        elif key == 45 : #-:
            if choose == BLUE:
                thresh_blue -= 2
            elif choose == RED:
                thresh_red -= 2

        elif key == 43 : #-:
            if choose == BLUE:
                thresh_blue += 2
            elif choose == RED:
                thresh_red += 2
        
        print('red:', thresh_red, '   blue:', thresh_blue )
        


