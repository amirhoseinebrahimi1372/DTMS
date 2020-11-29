import cv2
import numpy as np
import numba
from numba import cuda
from pypylon import pylon
import time

UP_LEFT = 0
UP_RIGHT = 1
DOWN_LEFT = 2
DOWN_RIGHT = 3
RED = 5
BLUE = 6
KEYS = [ 'UL','UR', 'DL', 'DR']

step_move = 5
step_rotate = 5

DEBUG_ORGINAL = 0
DEBUG_PERSPECTIVE = 1
DEBUG_LASER = 2
DEBUG_MERGE = 3

#_______________________________________________________________________________________________________________________________________________
#                                                       this Class Yield Frame of Camera
#_______________________________________________________________________________________________________________________________________________
class cameraGraber:
    def __init__(self, idx):
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        print( len( devices ))
        self.camera = pylon.InstantCamera()
        self.camera.Attach(tlFactory.CreateDevice(devices[idx]))
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        #--------------------------------------5-----------------------------------------------
        
        
        self.camera.GainAuto.SetValue('Off')
        self.camera.GainSelector.SetValue('All')
        self.camera.GainRaw.SetValue(10)
        self.camera.GammaSelector.SetValue('User')
        #cameras['ul'].camera.BlackLevelSelector.SetValue('All')

        #self.camera.LightSourceSelector.SetValue('Daylight 6500 Kelvin')
        self.camera.BalanceWhiteAuto.SetValue('Off')
        self.camera.ColorAdjustmentEnable.SetValue(False)
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.ExposureTimeRaw.SetValue(20000)
        self.camera.ExposureTimeAbs.SetValue(20000)
        
        #self.camera.GammaSelect #user
        #self.camera.BalanceRatioRaw.SetValue(130)
        #self.camera.LightSourceSelector.Set(4)
        #self.camera.TriggerMode.SetValue(1)
        
        #self.camera.AutoExposureTimeAbsLowerLimit.SetValue(0)
        #self.camera.BalanceRatioRaw.SetValue(135)
        #self.camera.ExposureTimeRaw.SetValue(3000)
        #self.camera.GammaSelector.SetIntValue(1)
        #self.camera.GainRaw.SetValue(158)
        
        #self.camera.DigitalShift.SetValue(3)
        # converting to opencv bgr format
        #cameras['ul'].camera.GainRaw.SetValue(80)
        
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        


    def grabber( self, rotate=None ):
        i=0
        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                img = image.GetArray()

                if rotate != None:
                    img = cv2.rotate( img, rotate )
                yield img

#________________________________________________________________________________________________________________________________________
#                                                           Main
#________________________________________________________________________________________________________________________________________
def getCamerasDict( ul=0, ur=0, dl=0, dr=0):
    cams={}
    cams['UL'] = cameraGraber(ul)
    cams['UR'] = cameraGraber(ur)
    cams['DL'] = cameraGraber(dl)
    cams['DR'] = cameraGraber(dr)
    return cams



#________________________________________________________________________________________________________________________________________
#                                                           Laser Detection
#________________________________________________________________________________________________________________________________________
@cuda.jit('void(uint8[:,:,:], uint8,uint8, uint8[:,:])')
def laser_detection_cuda( img, thresh,  color, result ):

    i, j = cuda.grid(2)
    img_rows, img_cols = img.shape[:2]
    
    if (i >= img_rows) or (j >= img_cols):
        return

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
    '''
    blockdim = (32, 32)
    griddim = (img.shape[0] // blockdim[0] + 1, img.shape[1] // blockdim[1] + 1)
    
    img_cuda = cuda.to_device(img)
    reslut = np.zeros( img.shape[:2], img.dtype )
    result_cuda = cuda.to_device(reslut)
    
    laser_detection_cuda[griddim, blockdim]( img_cuda, thresh, color, result_cuda)
    return result_cuda.copy_to_host()
    '''
    b,g,r = cv2.split(img)
    if color == RED:
        msk = r - np.maximum( g,b ).astype( np.int16)
        msk = np.clip(msk, 0, 255 ).astype( np.uint8)
        _, msk = cv2.threshold( msk, thresh, 255, cv2.THRESH_BINARY)# + cv2.THRESH_OTSU )
        msk = cv2.morphologyEx( msk, cv2.MORPH_OPEN, (3,3), iterations=2)
        
    
    return msk
#________________________________________________________________________________________________________________________________________
#                                                           Keyboard action
#________________________________________________________________________________________________________________________________________

def keyboard_listener( key, debug_flags ):
    global debug_camera_idx,merge_coordinate, step_move, angles, step_rotate
    if key < 0:
        return debug_flags

    
    elif key == 49: #1
        debug_camera_idx = 0
    elif key == 50: #2
        debug_camera_idx = 1
        cv2.destroyAllWindows()
    elif key == 51: #3
        debug_camera_idx = 2
        cv2.destroyAllWindows()
    elif key == 52: #4
        debug_camera_idx = 3
        cv2.destroyAllWindows()
    elif key == 53: #5
        debug_camera_idx = 4
        cv2.destroyAllWindows()
        
    elif key == 15 : #ctrl + O
        debug_flags = [False] * len(debug_flags)
        debug_flags[ DEBUG_ORGINAL ] = True
        cv2.destroyAllWindows()

    elif key == 12 : #ctrl + L
        debug_flags = [False] * len(debug_flags)
        debug_flags[ DEBUG_LASER ] = True
        cv2.destroyAllWindows()

    elif key == 16 : #ctrl + P
        debug_flags = [False] * len(debug_flags)
        debug_flags[ DEBUG_PERSPECTIVE ] = True
        cv2.destroyAllWindows()

    elif key == 13 : #ctrl + M
        debug_flags = [False] * len(debug_flags)
        debug_flags[ DEBUG_MERGE ] = True
        cv2.destroyAllWindows()



    if debug_flags[ DEBUG_PERSPECTIVE ]:

        if key == ord('q'):
            angles[ KEYS[ debug_camera_idx]][0] += step_rotate
        elif key == ord('a'):
            angles[ KEYS[ debug_camera_idx]][0] -= step_rotate
        elif key == ord('w'):
            angles[ KEYS[ debug_camera_idx]][1] += step_rotate
        elif key == ord('s'):
            angles[ KEYS[ debug_camera_idx]][1] -= step_rotate
        elif key == ord('e'):
            angles[ KEYS[ debug_camera_idx]][2] += step_rotate
        elif key == ord('d'):
            angles[ KEYS[ debug_camera_idx]][2] -= step_rotate

        elif key == 45: #-
            step_rotate -=1
            if step_rotate< 1:
                step_rotate=1

        elif key == 43: #+
            step_rotate +=1
            if step_rotate> 10:
                step_rotate=10
                
        print(angles, step_rotate)


    if debug_flags[ DEBUG_MERGE ]:
        
        if key == ord('a'):
            merge_coordinate[ KEYS[ debug_camera_idx]][0] -= step_move
        elif key == ord('d'):
            merge_coordinate[ KEYS[ debug_camera_idx]][0] += step_move
        elif key == ord('w'):
            merge_coordinate[ KEYS[ debug_camera_idx]][1] -= step_move
        elif key == ord('s'):
            merge_coordinate[ KEYS[ debug_camera_idx]][1] += step_move

        elif key == 45: #-
            step_move -=1
            if step_move< 1:
                step_move=1

        elif key == 43: #+
            step_move +=1
            if step_move> 10:
                step_move=10

        print(merge_coordinate, step_move)

        

    


    
    
    return debug_flags
        


#________________________________________________________________________________________________________________________________________
#                                                           Debug UI
#________________________________________________________________________________________________________________________________________
def debug_ui( debug_flags ):
    if debug_flags[ DEBUG_ORGINAL ]:
            if debug_camera_idx < 4:
                cv2.imshow( KEYS[ debug_camera_idx] + ' orginal' , cv2.resize(frames[ KEYS[ debug_camera_idx] ], None, fx=0.5, fy=0.5))                
            else :
                for key in KEYS:
                    cv2.imshow( key + ' orginal' , cv2.resize(frames[ key ], None, fx=0.5, fy = 0.5))

    if debug_flags[ DEBUG_PERSPECTIVE ]:
            pass

    if debug_flags[ DEBUG_LASER ]:
            if debug_camera_idx < 4:
                cv2.imshow( KEYS[ debug_camera_idx] + ' laser detection' , cv2.resize(laser_mask[ KEYS[ debug_camera_idx] ], None, fx=0.5, fy=0.5))                
            else :
                for key in KEYS:
                    cv2.imshow( key + ' perspective rotation' , cv2.resize(laser_mask[ key ], None, fx=0.5, fy = 0.5))

    if debug_flags[ DEBUG_MERGE ]:
            merge = np.copy( merge_img )
            cv2.putText(merge,'UP_LEFT',(70,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(merge,'UP_RIGHT',(500,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(merge,'DOWN_LEFT',(1000,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(merge,'DOWN_RIGHT',(1500,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0,127, 127),2,cv2.LINE_AA)

            if debug_camera_idx == 0 :
                cv2.circle(merge, (2000, 40),  20, (255,0,0), thickness=-1)

            if debug_camera_idx == 1 :
                cv2.circle(merge, (2000, 40),  20, (0,255,0), thickness=-1)

            if debug_camera_idx == 2 :
                cv2.circle(merge, (2000, 40),  20, (0,250,250), thickness=-1)

            if debug_camera_idx == 3 :
                cv2.circle(merge, (2000, 40),  20, (0,0,255), thickness=-1 )

            cv2.imshow( ' merge' , cv2.resize(merge, None, fx=0.5, fy = 0.5))




#________________________________________________________________________________________________________________________________________
#                                                             Main 
#________________________________________________________________________________________________________________________________________

debug_camera_idx = 0
debug_flags = [True, False, False, False, False]

th = 100
merge_coordinate = {'UL': [-265, -10], 'UR': [0, -5], 'DL': [-175, -295], 'DR': [70, -135]}


if __name__ == '__main__':

    
    
    cameras = getCamerasDict( ul=3,ur=1, dl=2,dr=0)
    while True:            
        frames={}
        frames['UL'] = next(cameras['UL'].grabber(rotate = None))
        frames['UR'] = next(cameras['UR'].grabber(rotate = None))
        frames['DL'] = next(cameras['DL'].grabber(rotate = None))
        frames['DR'] = next(cameras['DR'].grabber(rotate = None))

        #------------------------------------------------------------------
        #------------------------------------------------------------------
        laser_mask={}
        laser_mask['UL'] = laser_detection( frames['UL'],140, RED )
        laser_mask['UR'] = laser_detection( frames['UR'],140, RED )
        laser_mask['DL'] = laser_detection( frames['DL'],140, RED )
        laser_mask['DR'] = laser_detection( frames['DR'],140, RED )
        
        #------------------------------------------------------------------
        #------------------------------------------------------------------

        BOX_SIZE=100#mm
        BORDER = 20
        for i in range(len(KEYS)):
            corners = []
            msk = laser_mask[ KEYS[i] ]
            pts = np.argwhere(msk)
            pts = np.array([pts[:,1],pts[:,0]]).transpose()
            pts = list(pts)
            pts.sort( key= lambda x:x[0])
            corners.append( pts[0] )
            corners.append( pts[-1] )
            pts.sort( key= lambda x:x[1])
            corners.append( pts[0] )

            if KEYS[i] == 'UL':
                corners.sort( key= lambda x:x[0])
                for num,pt in enumerate(corners):
                    cv2.circle( frames[ KEYS[i]], tuple(pt), 5, (0,255,0), thickness=-1)
                    cv2.putText( frames[ KEYS[i]] , str(num), tuple(pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))

                cv2.imshow('frame_' + KEYS[i], cv2.resize(frames[ KEYS[i]], None, fx=0.5, fy=0.5) )

                start_pts = np.array( corners, dtype = np.float32 )
                dest_pts = np.array([[BORDER,BORDER], [BORDER + BOX_SIZE,BORDER], [BORDER + BOX_SIZE,BORDER + BOX_SIZE]], dtype = np.float32)

                M = cv2.getAffineTransform(start_pts,dest_pts )
                while True:
                    img = next(cameras[KEYS[i]].grabber(rotate = None))
                    wrap_img = cv2.warpAffine( img, M,img.shape[0:2]) #(2 * BORDER + BOX_SIZE, 2*BORDER + BOX_SIZE))
                    cv2.imshow('wrap img', wrap_img)
                    cv2.imshow('img', img)
                    cv2.waitKey(30)
                #cv2.waitKey(0)
                
            
            
            

        debug_ui( debug_flags )
        keyboard = cv2.waitKey(100)
        
        print(keyboard)
        debug_flags = keyboard_listener( keyboard, debug_flags )

