import cv2
import numpy as np
import numba
from numba import cuda
from pypylon import pylon
import time
import os

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
#                                                             Main 
#________________________________________________________________________________________________________________________________________



def chessPerspectiveCalib(camera, cam_pos, chess_size = (9,6), chess_home_size = 30, border = 2):
    while True:
        img = next(camera.grabber(rotate=None))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size,None)
        if not ret:
            cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(100)
            continue
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners = corners.reshape((-1,2))

        org_pts = []
        if cam_pos == 'UL':
            org_pts.append( corners[0] )
            org_pts.append( corners[chess_size[0] - 1] )
            org_pts.append( corners[chess_size[0]*chess_size[1] - chess_size[0]] )
            org_pts.append( corners[-1] )
            
            for num,corner in enumerate(org_pts):
                cv2.circle( img, tuple(corner), 5, (0,0,255), thickness=-1)
                cv2.putText( img , str(num), tuple(corner), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255), thickness=2)
                
            cv2.imshow('image', cv2.resize( img, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(0)
            if key==ord('n'):
                continue

            
            org_pts = np.array( org_pts, dtype = np.float32)
            dst_pts = np.array( [[ border * chess_home_size,      border * chess_home_size],
                                 [(border + chess_size[0]-1)* chess_home_size ,      border * chess_home_size],
                                 [ border * chess_home_size, (border + chess_size[1]-1)* chess_home_size  ],
                                 [(border + chess_size[0]-1)* chess_home_size, (border + chess_size[1]-1)* chess_home_size]]).astype(np.float32 )
            M = cv2.getPerspectiveTransform( org_pts, dst_pts)
            
            #dst_h = (2*border + chess_size[1]-1)* chess_home_size
            #dst_w = (2*border + chess_size[0]-1)* chess_home_size
            dst_w, dst_h = (2*border + chess_size[0]-1)* chess_home_size, (2*border + chess_size[1]-1)* chess_home_size

            while True:
                dst = cv2.warpPerspective( img, M, (dst_w, dst_h))

                cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5 ))
                cv2.imshow('dst', cv2.resize(dst, None, fx=0.5, fy=0.5 ))
                key = cv2.waitKey(50)
                if key == ord('n') or key == 27 or  key == ord('y'):
                    cv2.destroyAllWindows()
                    break
            if key == 27 or key == ord('y'):
                break


class calibration_perspective():

    def __init__(self, path, chess_size = (9,6), chess_home_size = 30, border = 2, cam_positions = ['UL', 'UR', 'DL', 'DR'] ):
        self.chess_size = chess_size
        self.chess_home_size = chess_home_size
        self.chess_home_size_px = self.chess_home_size 
        self.border = border
        self.cam_positions = cam_positions
        self.path = path
        self.mtx= {}
        self.dst_size=[]
        self.px2mm = 1




    def calc_accuracy(self, cameras, debug=True):
        distances = []
        for cam_pose,cam in cameras.items():
            while True:
                img = next( cam.grabber() )
                #img = cv2.imread('data/cal_img/cal_1.jpg')
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                print('Search For Chess')
                ret, corners = cv2.findChessboardCorners(gray, self.chess_size,None)
                
                if debug:
                    cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))   
                    cv2.waitKey(10)
                if not ret:
                    print('Not Find')
                    continue
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                corners = corners.reshape((-1,2))

                
                for i in range(len( corners) - self.chess_size[0]):
                    pt1 = corners[i]
                    pt2 = corners[i+1]
                    pt3 = corners[i+self.chess_size[0]]

                    l1 = int( ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5 )
                    l2 = int( ((pt1[0] - pt3[0])**2 + (pt1[1] - pt3[1])**2)**0.5 )
                    
                    distances.append(l1)
                    distances.append(l2)


                    if debug:
                        res = np.copy(img)
                        cv2.circle( res, tuple(pt1), 2, (0,0,255), thickness=-1)
                        cv2.circle( res, tuple(pt2), 2, (0,0,255), thickness=-1)
                        cv2.circle( res, tuple(pt3), 2, (0,0,255), thickness=-1)

                        cv2.putText( res , str(l1), tuple(((pt1 + pt2)/2 + (-10,-10)).astype(np.int32)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,0,255), thickness=2)
                        cv2.putText( res , str(l2), tuple(((pt1 + pt3)/2 + (-20,5)).astype(np.int32)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0,0,255), thickness=2)
                            
                        cv2.imshow('image', cv2.resize( res, None, fx=0.75, fy=0.75))
                        key = cv2.waitKey(10)
                        if key == 27:
                            break
                break

        distances = np.array([distances])
        if debug:
            cv2.destroyAllWindows()
            print( 'min px size: ', distances.min())
            print( 'accuracy: ',self.chess_home_size / distances.min(),'mm')
        self.chess_home_size_px = distances.min()
        self.px2mm = self.chess_home_size / distances.min()
        #np.savenp.array([self.chess_home_size / distances.min()])
        return distances.min(), self.chess_home_size / distances.min()




    
    def calc_calibration(self, cameras, debug=True):

        for cam_pos,cam in cameras.items():
            while True:
                img = next( cam.grabber() )
                #img = cv2.imread('data/cal_img/cal_1.jpg')
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.chess_size,None)
                #----------------------------------
                if debug:
                    cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))
                    cv2.waitKey(10)
                if not(ret):
                    continue
                cv2.destroyAllWindows()
                #---------------------------------- 
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                corners = corners.reshape((-1,2))
                #----------------------------------
                if debug:
                    img_ = np.copy(img)
                    for num,corner in enumerate(corners):
                        cv2.circle( img_, tuple(corner), 5, (0,0,255), thickness=-1)
                        cv2.putText( img_ , str(num), tuple(corner), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), thickness=2)
                        
                    cv2.imshow('image_all_corners_' + cam_pos, cv2.resize( img_, None, fx=0.5, fy=0.5))
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if key==ord('n'):
                        continue
                #----------------------------------
                org_pts = []
                if cam_pos == 'UL':
                    org_pts.append( corners[0] )
                    org_pts.append( corners[self.chess_size[0] - 1] )
                    org_pts.append( corners[self.chess_size[0] * self.chess_size[1] - self.chess_size[0]] )
                    org_pts.append( corners[-1] )


                if cam_pos == 'UR':
                    org_pts.append( corners[self.chess_size[0] * self.chess_size[1] - self.chess_size[0]] )
                    org_pts.append( corners[0] )
                    org_pts.append( corners[-1] )
                    org_pts.append( corners[self.chess_size[0] - 1] )



                if cam_pos == 'DL':
                    org_pts.append( corners[self.chess_size[0] - 1] )
                    org_pts.append( corners[-1] )
                    org_pts.append( corners[0] )
                    org_pts.append( corners[self.chess_size[0] * self.chess_size[1] - self.chess_size[0]] )
                    


                if cam_pos == 'DR':
                    org_pts.append( corners[-1] )
                    org_pts.append( corners[self.chess_size[0] * self.chess_size[1] - self.chess_size[0]] )
                    org_pts.append( corners[self.chess_size[0] - 1] )
                    org_pts.append( corners[0] )
                    
                    
                    

                    

                if debug:
                    img_ = np.copy(img)
                    for num,corner in enumerate(org_pts):
                        cv2.circle( img_, tuple(corner), 5, (0,0,255), thickness=-1)
                        cv2.putText( img_ , str(num), tuple(corner), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255), thickness=2)
                        
                    cv2.imshow('image_border_corners_' + cam_pos, cv2.resize( img_, None, fx=0.5, fy=0.5))
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if key==ord('n'):
                        continue
                
                #----------------------------------
                org_pts = np.array( org_pts, dtype = np.float32)
                dst_pts = np.array( [[ self.border * self.chess_home_size_px,      self.border * self.chess_home_size_px],
                                     [(self.border + self.chess_size[0]-1)* self.chess_home_size_px ,      self.border * self.chess_home_size_px],
                                     [ self.border * self.chess_home_size_px, (self.border + self.chess_size[1]-1)* self.chess_home_size_px  ],
                                     [(self.border + self.chess_size[0]-1)* self.chess_home_size_px, (self.border + self.chess_size[1]-1)* self.chess_home_size_px]]).astype(np.float32 )
                #----------------------------------
                M = cv2.getPerspectiveTransform( org_pts, dst_pts)
                dst_w, dst_h = (2*self.border + self.chess_size[0]-1)* self.chess_home_size_px, (2*self.border + self.chess_size[1]-1)* self.chess_home_size_px
                self.dst_size = dst_w, dst_h
                np.save( os.path.join(self.path,'dsize'),  np.array([dst_w, dst_h] ))
                np.save( os.path.join(self.path,'mtx_' + cam_pos ), M)

                
                if debug:
                    while True:
                        img = next( cam.grabber())
                        dst = cv2.warpPerspective( img, M, (dst_w, dst_h))
                        cv2.imshow('image', cv2.resize(img, None, fx=0.75, fy=0.75 ))
                        cv2.imshow('dst', cv2.resize(dst, None, fx=0.75, fy=0.75 ))
                        key = cv2.waitKey(50)
                        if key == ord('n') or key == 27 or  key == ord('y'):
                            cv2.destroyAllWindows()
                            break
                    if key == 27 or key == ord('y'):
                        break
                         
        
    def load_mtx(self, cameras):
        for cam_pos,cam in cameras.items():
            try:
                self.mtx[cam_pos] = np.load( os.path.join( self.path, 'mtx_' + cam_pos + '.npy') )
            except:
                print('mtx_' + cam_pos + ' not found')

            try:
                self.dst_size = np.load('data/perspective_calib/dsize.npy')
            except:
                print('dst_size not found')
        

    def correct_all_persective(self, imgs):
        if len(self.dst_size)==0 or len(imgs)>len(self.mtx):
            print('matrix or dst_size not loaded')
            return -1
        ress={}
        for cam_pos,img in imgs.items():
            ress[cam_pos] =  cv2.warpPerspective( img, self.mtx[cam_pos], tuple(self.dst_size))
        return ress

    def correct_persective(self, img, cam_pos):
        if len(self.dst_size)==0 or self.mtx.get( cam_pos ) is None:
            print('matrix or dst_size not loaded')
            return -1
        res =  cv2.warpPerspective( img, self.mtx[cam_pos], tuple(self.dst_size))
        return res
            


if __name__ == '__main__':

    cameras = getCamerasDict( ul=1,ur=3, dl=0,dr=2)
    #cameras = {'UL':5, 'DL':30,'UR':1}
    
    perspective = calibration_perspective('data/perspective_calib', chess_size = (6,6), chess_home_size = 20, border = 3)
    perspective.calc_accuracy( cameras )
    perspective.calc_calibration( cameras )
    perspective.load_mtx(cameras)




    
    '''

    ret = np.load(path_cal + '/ret.npy')
    mtx = np.load(path_cal + '/K.npy')
    dist = np.load(path_cal + '/dist.npy')
    rvecs = np.load(path_cal + '/rvecs.npy')
    tvecs = np.load(path_cal + '/tvecs.npy')


    cam_pos = 'UL'
    chess_size = (9,6)
    chess_home_size = 30
    border = 2
    while True:
        img = cv2.imread('data/cal_img/cal_18.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size,None)
        if not ret:
            cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(100)
            continue
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners = corners.reshape((-1,2))

        org_pts = []
        if cam_pos == 'UL':
            org_pts.append( corners[0] )
            org_pts.append( corners[chess_size[0] - 1] )
            org_pts.append( corners[chess_size[0]*chess_size[1] - chess_size[0]] )
            org_pts.append( corners[-1] )
            
            for num,corner in enumerate(org_pts):
                cv2.circle( img, tuple(corner), 5, (0,0,255), thickness=-1)
                cv2.putText( img , str(num), tuple(corner), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,0,255), thickness=2)
                
            cv2.imshow('image', cv2.resize( img, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(0)
            if key==ord('n'):
                continue

            
            org_pts = np.array( org_pts, dtype = np.float32)
            dst_pts = np.array( [[ border * chess_home_size,      border * chess_home_size],
                                 [(border + chess_size[0]-1)* chess_home_size ,      border * chess_home_size],
                                 [ border * chess_home_size, (border + chess_size[1]-1)* chess_home_size  ],
                                 [(border + chess_size[0]-1)* chess_home_size, (border + chess_size[1]-1)* chess_home_size]]).astype(np.float32 )
            M = cv2.getPerspectiveTransform( org_pts, dst_pts)
            
            #dst_h = (2*border + chess_size[1]-1)* chess_home_size
            #dst_w = (2*border + chess_size[0]-1)* chess_home_size
            dst_w, dst_h = (2*border + chess_size[0]-1)* chess_home_size, (2*border + chess_size[1]-1)* chess_home_size

            while True:
                dst = cv2.warpPerspective( img, M, (dst_w, dst_h))

                cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5 ))
                cv2.imshow('dst', cv2.resize(dst, None, fx=0.5, fy=0.5 ))
                key = cv2.waitKey(50)
                if key == ord('n') or key == 27 or  key == ord('y'):
                    cv2.destroyAllWindows()
                    break
            if key == 27 or key == ord('y'):
                break
                
                
            
                            
                


        cv2.imshow('img', cv2.resize( img, None, fx=0.5, fy=0.5))
        cv2.waitKey(50)
    '''

            


                    
                    
                





        




