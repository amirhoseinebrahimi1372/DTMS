import cv2
import numpy as np
import numba
from numba import cuda
from pypylon import pylon
import time
import os
import perspectiveCalbration as pc
import cameraCalibration as cc


KEYS = [ 'UL','UR', 'DL', 'DR']



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
    delay = 0.5
    cams['UL'] = cameraGraber(ul)
    time.sleep(delay)
    cams['UR'] = cameraGraber(ur)
    time.sleep(delay)
    cams['DL'] = cameraGraber(dl)
    time.sleep(delay)
    cams['DR'] = cameraGraber(dr)
    time.sleep(delay)
    return cams




class merge_calibragtion():



    def __init__(self,path):
        self.mrg_pts={}
        self.path = path
        self.res_shape = ()



    def calibration(self,  imgs, w,h, px_mm = 10, search = 10, border = 2, debug = True ):
        
        w_px = int(w / px_mm)
        h_px = int(h / px_mm)

        ptrn_shape = imgs['UL'].shape#[0] + search * border * 2, imgs['UL'].shape[1] + search * border * 2
        self.res_shape = ptrn_shape
        np.save( os.path.join(self.path, 'dst_size'), np.array(ptrn_shape))
        pattern = np.zeros( ptrn_shape , np.uint8 )
        #-----------------------------------
        pt1 = int((pattern.shape[1] - w_px)/2), int((pattern.shape[0] - h_px)/2) 
        cv2.rectangle( pattern, pt1, ( pt1[0] + w_px, pt1[1] + h_px), 255 )
        if debug:
            cv2.imshow('pattern',cv2.resize(pattern, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #-----------------------------------
        for cam_pos,img in imgs.items():
            points_img = np.argwhere(img)
            points_img = np.array([points_img[:,1],points_img[:,0]]).transpose()
            points_pattern = np.argwhere(pattern)
            points_pattern = np.array([points_pattern[:,1],points_pattern[:,0]]).transpose()

            #---------------------------------------
            if cam_pos == 'UL':
                x = points_pattern[:,0].min() - points_img[:,0].min()
                y = points_pattern[:,1].min() - points_img[:,1].min()

            elif cam_pos == 'UR':
                x = points_pattern[:,0].max() - points_img[:,0].max()
                y = points_pattern[:,1].min() - points_img[:,1].min()

            elif cam_pos == 'DL':
                x = points_pattern[:,0].min() - points_img[:,0].min()
                y = points_pattern[:,1].max() - points_img[:,1].max()

            elif cam_pos == 'DR':
                x = points_pattern[:,0].max() - points_img[:,0].max()
                y = points_pattern[:,1].max() - points_img[:,1].max()

            #---------------------------------------
            if debug:
                print('Search Start')
            search_resluts = []
            for x_ in range(x-search, x + search):
                for y_ in range(y-search, y + search):
                    res = self.add( pattern, img, x_,y_ )
                    sum_pattern = np.sum( pattern )
                    sum_img = np.sum( img )
                    sum_res = np.sum( res )
                    error = sum_res/( sum_img + sum_pattern )
                    search_resluts.append( [error, x_, y_])
                    
            #---------------------------------------
            search_resluts.sort( key = lambda e:e[0] )
            search_resluts = search_resluts[0][1:]
            search_resluts = np.array(search_resluts)
            np.save( os.path.join( self.path, 'pts_' + cam_pos ), search_resluts )
            self.mrg_pts[ cam_pos ] = search_resluts

            if debug:
                res = self.add( pattern, img, search_resluts[0],search_resluts[1] )
                cv2.imshow('res', cv2.resize(res, None, fx=0.5, fy=0.5))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                    
        
        
        
    def add(self, merge_img, img, x, y):
        res = np.copy( merge_img )
        if x < 0:
            x_out = 0
            x_in = abs(x)
        else :
            x_out = x
            x_in = 0

        if y < 0:
            y_out = 0
            y_in = abs(y)
        else :
            y_out = y
            y_in = 0
        
        h = np.minimum( img.shape[0] - y_in , res.shape[0] - y_out )
        w = np.minimum( img.shape[1] - x_in , res.shape[1] - x_out )

        res[ y_out: y_out + h , x_out: x_out + w ] = cv2.add(res[ y_out: y_out + h ,  x_out: x_out + w ], img[ y_in: y_in + h ,  x_in: x_in + w  ] )

        return res



    def load(self):
        for cam_pos in ['UL','UR','DL','DR']:
            try:
                self.mrg_pts[cam_pos] = np.load( os.path.join( self.path, 'pts_' + cam_pos + '.npy'))
            except:
                print('ERROR:not found')

            self.res_shape = np.load( os.path.join( self.path, 'dst_size.npy'))



    def merge(self, img, cam_pos, res= None):
        if len(self.res_shape)==0 or self.mrg_pts.get(cam_pos) is None:
            print('Result shape Or Cordinate Not Find')
            return -1
        if res is None:
            res = np.zeros( self.res_shape, np.uint8 )
        return self.add( res, img, self.mrg_pts[cam_pos][0], self.mrg_pts[cam_pos][1] )



            


    
if __name__ == '__main__':

    '''
    imgs={}
    imgs['UL'] = cv2.imread('ul.png',0)
    imgs['UR'] = cv2.imread('ur.png',0)
    imgs['DL'] = cv2.imread('dl.png',0)
    imgs['DR'] = cv2.imread('dr.png',0)

    mrg = merge_calibragtion('data/merge_calib')
    #mrg.calibration(imgs, 20,10,px_mm = 0.02 )
    mrg.load()
    t = time.time()
    res = mrg.merge( imgs['UL'], 'UL' )
    res = mrg.merge( imgs['UR'], 'UR', res )
    res = mrg.merge( imgs['DL'], 'DL', res )
    res = mrg.merge( imgs['DR'], 'DR', res )
    print(t)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    '''
    cameras = getCamerasDict( ul=1,ur=3, dl=0,dr=2)
    camera_calibration = cc.CameraCalibration('data/camera_calib')
    calib_perspective = pc.calibration_perspective('data/perspective_calib', camera_calibration, chess_size = (6,6), chess_home_size = 20, border = 3)
    calib_perspective.load_mtx(cameras)

    chess_size = (6,6)
    res = 0
    for cam_pos, cam in cameras.items():
        #while True:
        img = next( cam.grabber() )
        img = calib_perspective.correct_persective(img, cam_pos )
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        res = res + img.astype(np.int64)
    
    res = res / len(cameras)
    res= res.astype(np.uint8)
    cv2.imshow('res', res )
    cv2.waitKey(0)
        
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chess_size,None)
    if not ret:

    
        cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.waitKey(100)
        continue
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    corners = corners.reshape((-1,2))
    if chess_size[0]%2 == 0:
        x =  int(corners[ chess_size[0]/2 , 0] +  corners[ chess_size[0]/2 - 1 , 0] ) ) 
    else :
        x =  int(corners[ int(chess_size[0]/2) , 0]

    if chess_size[1]%2 == 0:
        y =  int(corners[ chess_size[0] , 1] +  corners[ chess_size[0]/2 - 1 , 0] ) ) 
    else :
        y =  int(corners[ int(chess_size[0]/2) , 0]
    '''
                         
                

        
        
    

    


            



        
                    
                





        




