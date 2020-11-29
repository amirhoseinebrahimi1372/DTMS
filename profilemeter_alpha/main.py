import cv2
import time
import numba
import numpy as np
from numba import cuda
from pypylon import pylon
from  sklearn import  cluster
import cameraCalibration as cc
import perspectiveCalbration as pc
from matplotlib import pyplot as plt


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








'''
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
    camera.Width.SetValue(new_width)

numberOfImagesToGrab = 100
camera.StartGrabbingMax(numberOfImagesToGrab)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        print("Gray value of first pixel: ", img[0, 0])

    grabResult.Release()
camera.Close()
'''
#_______________________________________________________________________________________________________________________________________________
#                                                       this Class Yield Frame of Camera
#_______________________________________________________________________________________________________________________________________________
class cameraGraber:
    def __init__(self, idx):
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        else:
            print('Numbers Of Camera:', len( devices ))

        #cameras = pylon.InstantCameraArray(len(devices))
        self.camera = pylon.InstantCamera()
        self.camera.Attach(tlFactory.CreateDevice(devices[idx]))
        self.camera.StartGrabbing()#pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        #--------------------------------------5-----------------------------------------------
        
        
        self.camera.GainAuto.SetValue('Off')
        self.camera.GainSelector.SetValue('All')
        self.camera.GainRaw.SetValue(100)
        self.camera.GammaSelector.SetValue('User')
        #cameras['ul'].camera.BlackLevelSelector.SetValue('All')

        #self.camera.LightSourceSelector.SetValue('Daylight 6500 Kelvin')
        self.camera.BalanceWhiteAuto.SetValue('Off')
        self.camera.ColorAdjustmentEnable.SetValue(False)
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.ExposureTimeRaw.SetValue(1000)
        self.camera.ExposureTimeAbs.SetValue(1000)
        
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

                #if rotate != None:
                #    img = cv2.rotate( img, rotate )
                yield img

#________________________________________________________________________________________________________________________________________
#                                                           Main
#________________________________________________________________________________________________________________________________________
def getCamerasDict( ul=0, ur=0, dl=0, dr=0):
    cams={}
    delay = 0.2
    cams['UL'] = cameraGraber(ul)
    time.sleep(delay)
    cams['UR'] = cameraGraber(ur)
    time.sleep(delay)
    cams['DL'] = cameraGraber(dl)
    time.sleep(delay)
    cams['DR'] = cameraGraber(dr)
    time.sleep(delay)
    return cams
#________________________________________________________________________________________________________________________________________
#                                                           Wrapp Imgage
#________________________________________________________________________________________________________________________________________
def wrapp(img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        h,w = img.shape[0:2]
        d = np.sqrt(h**2 + w**2)
        
        rtheta, rphi, rgamma = np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(gamma) 
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(rtheta), -np.sin(rtheta), 0],
                        [0, np.sin(rtheta), np.cos(rtheta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(rphi), 0, -np.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [np.sin(rphi), 0, np.cos(rphi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(rgamma), -np.sin(rgamma), 0, 0],
                        [np.sin(rgamma), np.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [focal, 0, w/2, 0],
                        [0, focal, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        mtx =  np.dot(A2, np.dot(T, np.dot(R, A1)))
        
        return cv2.warpPerspective(img.copy(), mtx, (w, h))
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

    blockdim = (32, 32)
    griddim = (img.shape[0] // blockdim[0] + 1, img.shape[1] // blockdim[1] + 1)
    
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



def laserDetectionPro( img, thresh=120, eps=10, min_samples=5 ):
    b,g,r = cv2.split( img )
    msk = r - np.maximum(b,g).astype( np.int16 )
    msk = np.clip(msk, 0, 255 ).astype(np.uint8 )
    _, msk = cv2.threshold( msk , thresh , 255  , cv2.THRESH_BINARY )
    pts = np.transpose( np.nonzero(msk) )
    if len(pts)<50:
        return -1
    clst = cluster.DBSCAN( eps=eps, min_samples = min_samples ).fit(pts )
    lbl = clst.labels_
    lbl[lbl<0] = lbl.max()+1
    hist = np.bincount( lbl )
    biggest_class = np.argmax( hist )
    res_pts = pts[lbl == biggest_class ]
    res_msk = np.zeros_like(msk)
    res_msk[ res_pts[:,0], res_pts[:,1]] = 255
    return res_msk

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
            if debug_camera_idx < 4:
                cv2.imshow( KEYS[ debug_camera_idx] + ' prspective' , cv2.resize(perspective[ KEYS[ debug_camera_idx] ], None, fx=0.5, fy=0.5))                
            else :
                for key in KEYS:
                    cv2.imshow( key + ' perspective rotation' , cv2.resize(perspective[ key ], None, fx=0.5, fy = 0.5))

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
debug_flags = [False, True, False, False, False]

th = 100
merge_coordinate = {'UL': [-265, -10], 'UR': [0, -5], 'DL': [-175, -295], 'DR': [70, -135]}
angles = {'UL':[0,-45,50], 'UR':[0,-40,-42], 'DL':[0,-45, 142], 'DR':[0,-45,230]}
angles = {'UL':[-45,-45,45], 'UR':[0,45,0], 'DL':[0,-45, 0], 'DR':[0,-35,0]}






if __name__ == '__main__':

    
    
    cameras = getCamerasDict( ul=3,ur=1, dl=0,dr=2)

    #cameras = {}
    #cameras['UL'] = cameraGraber(3)
    #time.sleep(delay)
    #cameras['UR'] = cameraGraber(1)
    #time.sleep(delay)
    #cameras['DL'] = cameraGraber(0)
    #time.sleep(delay)
    #cameras['DR'] = cameraGraber(dr)
    #time.sleep(delay)
    
    camera_calibration = cc.CameraCalibration('data/camera_calib')
    calib_perspective = pc.calibration_perspective('data/perspective_calib', camera_calibration, chess_size = (6,6), chess_home_size = 20, border = 3)
    calib_perspective.load_mtx(cameras)
    
    while True:

        
        delay = 0.00  
        frames={}
        for cam_pos, cam in cameras.items():
            frames[cam_pos] = next(cam.grabber(rotate = None))
        
        
        perspective = {}
        for cam_pos, img in frames.items():
            perspective[cam_pos] = calib_perspective.correct_persective(img, cam_pos )
            laser = np.zeros( perspective[cam_pos].shape[0:2], np.uint8 )
            cv2.imshow(cam_pos, cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.waitKey(30)            

        res = 0        
        for cam_pos, img in perspective.items():
            
            msk = laserDetectionPro( img, 25, eps=10, min_samples=5 )
            #cv2.imshow('img', img )

            laser = cv2.add( msk, laser)
            res = res + img.astype(np.int64)
        
        res = res / len(perspective)
        res= res.astype(np.uint8)

        laserDetectionPro( res, 120, eps=10,min_samples=5)
        #cv2.imshow('res', res )
        plt.imshow(msk)
        plt.pause(0.001)
        cv2.imshow('laser', laser )
        keyboard = cv2.waitKey(50)
        
        debug_ui( debug_flags )
        keyboard = cv2.waitKey(100)
        
        print(keyboard)
        debug_flags = keyboard_listener( keyboard, debug_flags )
        

        '''

        #------------------------------------------------------------------
        #------------------------------------------------------------------

        perspective_rotation={}
        perspective_rotation['UL'] = wrapp( frames['UL'], angles['UL'][0], angles['UL'][1],0 )
        perspective_rotation['UL'] = wrapp( perspective_rotation['UL'],0 , 0, angles['UL'][2] )
        
        perspective_rotation['UR'] = wrapp( frames['UR'], 0, angles['UR'][1], 0 )
        perspective_rotation['UR'] = wrapp( perspective_rotation['UR'], 0, 0, angles['UR'][2] )

        
        perspective_rotation['DL'] = wrapp( frames['DL'], 0, angles['DL'][1], 0)
        perspective_rotation['DL'] = wrapp( perspective_rotation['DL'], 0, 0, angles['DL'][2])
        
        perspective_rotation['DR'] = wrapp( frames['DR'], angles['DR'][0], angles['DR'][1], angles['DR'][2] )
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        laser_mask={}
        laser_mask['UL'] = laser_detection( perspective_rotation['UL'],140, RED )
        laser_mask['UR'] = laser_detection( perspective_rotation['UR'],140, RED )
        laser_mask['DL'] = laser_detection( perspective_rotation['DL'],140, RED )
        laser_mask['DR'] = laser_detection( perspective_rotation['DR'],140, RED )
        
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        merge_img = np.zeros( frames['UL'].shape, frames['UL'].dtype )

        for key in KEYS:
            if merge_coordinate[ key][0] < 0:
                x_out = 0
                x_in = abs( merge_coordinate[ key][0]  )
            else :
                x_out = merge_coordinate[ key][0]
                x_in = 0

            if merge_coordinate[ key][1] < 0:
                y_out = 0
                y_in = abs( merge_coordinate[ key][1]  )
            else :
                y_out = merge_coordinate[ key][1] 
                y_in = 0
            
            h_ = np.minimum( laser_mask[key].shape[0] - y_in , merge_img.shape[0] - y_out )
            w_ = np.minimum( laser_mask[key].shape[1] - x_in , merge_img.shape[1] - x_out )

            if key == 'UL':
                merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 0 ] = cv2.add(merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 0 ],
                                                                                  np.copy( laser_mask[key][ y_in: y_in + h_ ,  x_in: x_in + w_  ] ))
            if key == 'UR':
                merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 1 ] = cv2.add(merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 1 ],
                                                                                  np.copy( laser_mask[key][ y_in: y_in + h_ ,  x_in: x_in + w_  ] ))
            if key == 'DL':
                merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 2 ] = cv2.add(merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 2 ],
                                                                                  np.copy( laser_mask[key][ y_in: y_in + h_ ,  x_in: x_in + w_  ] ))
            if key == 'DR':
                merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 1 ] = cv2.add(merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 1 ],
                                                                                  (np.copy( laser_mask[key][ y_in: y_in + h_ ,  x_in: x_in + w_  ] )/2).astype(np.uint8))

                merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 2 ] = cv2.add(merge_img[ y_out: y_out + h_ ,  x_out: x_out + w_ , 2 ],
                                                                                  (np.copy( laser_mask[key][ y_in: y_in + h_ ,  x_in: x_in + w_  ] )/2).astype(np.uint8))


            

        

        #------------------------------------------------------------------

        #------------------------------------------------------------------
        '''
        

        







        
        
    
    '''
    img = cv2.imread('color_test.png')

    blockdim = (32, 32)
    griddim = (img.shape[0] // blockdim[0] + 1, img.shape[1] // blockdim[1] + 1)

    reslut = np.zeros( img.shape[:2] , dtype = np.uint8)
    t = time.time()
    laser_detection_cuda[griddim, blockdim]( img, 100, BLUE, reslut)
    print( time.time() - t )

    t = time.time()
    laser_detection_cuda[griddim, blockdim]( img, 100, RED, reslut)
    print( time.time() - t )


    t = time.time()
    reslut = laser_detection( img, 150, RED)
    reslut = laser_detection( img, 150, RED)
    reslut = laser_detection( img, 150, RED)
    reslut = laser_detection( img, 150, RED)
    print( 'n:',time.time() - t )


    




    cv2.imshow( ' res0', cv2.resize( reslut, None, fx=0.25, fy = 0.25 ))
    cv2.imshow( ' img', cv2.resize( img, None, fx=0.25, fy = 0.25 ))
    cv2.waitKey(5)
    '''
