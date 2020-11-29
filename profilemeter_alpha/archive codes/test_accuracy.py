import cv2
import numpy as np
import numba
from numba import cuda
from pypylon import pylon
import time
import os

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
        print( len( devices ),'a')
        self.camera = pylon.InstantCamera()
        self.camera.Attach(tlFactory.CreateDevice(devices[idx]))
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        #--------------------------------------5-----------------------------------------------
        
        
        self.camera.GainAuto.SetValue('Off')
        self.camera.GainSelector.SetValue('All')
        self.camera.GainRaw.SetValue(150)
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
#                                                           Wrapp Imgage
#________________________________________________________________________________________________________________________________________
def wrapp(img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, point = None):
        if point is None:
            point=[0,0]
            point[1] = img.shape[0]
            point[0] = img.shape[1]
            
        h,w = img.shape[0:2]
        d = np.sqrt(h**2 + w**2)
        
        rtheta, rphi, rgamma = np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(gamma) 
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -point[0]/2],
                        [0, 1, -point[1]/2],
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
        A2 = np.array([ [focal, 0, point[0]/2, 0],
                        [0, focal, point[1]/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        mtx =  np.dot(A2, np.dot(T, np.dot(R, A1)))
        
        return cv2.warpPerspective(img.copy(), mtx, (w, h))
#________________________________________________________________________________________________________________________________________
#                                                             Main 
#________________________________________________________________________________________________________________________________________

                                       
'''        
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img
'''
def draw1(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    deg = []
    for i in range(3):
        diff = corner - imgpts[i].ravel()
        deg.append(np.rad2deg(np.arctan( diff[1]/diff[0] )))
        
    
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img, deg



imgs_path = 'data\cal_img'
path_cal = 'data/calib'
deg = [0,0,0]
corners2=[[[0,0]]]



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
                
                
            


    
if __name__ == '__main__':


        ret = np.load(path_cal + '/ret.npy')
        mtx = np.load(path_cal + '/K.npy')
        dist = np.load(path_cal + '/dist.npy')
        rvecs = np.load(path_cal + '/rvecs.npy')
        tvecs = np.load(path_cal + '/tvecs.npy')


        cam_pos = 'UL'
        chess_size = (9,6)
        chess_home_size = 35
        distances = []
        for cam_pos in KEYS:
            while True:
                img = cv2.imread('data/cal_img/cal_13.jpg')
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chess_size,None)
                cv2.imshow('image', cv2.resize(img, None, fx=0.5, fy=0.5))
                cv2.waitKey(100)
                if not ret:
                    continue
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners= cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                corners = corners.reshape((-1,2))

                
                for i in range(len( corners) - chess_size[0]):
                    res = np.copy(img)
                    pt1 = corners[i]
                    pt2 = corners[i+1]
                    pt3 = corners[i+chess_size[0]]

                    l1 = int( ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5 )
                    l2 = int( ((pt1[0] - pt3[0])**2 + (pt1[1] - pt3[1])**2)**0.5 )
                    
                    distances.append(l1)
                    distances.append(l2)
                    
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
        cv2.destroyAllWindows()
        print( 'min px size: ', distances.min())
        print( 'accuracy: ',chess_home_size / distances.min(),'mm')
                    
                


            


                    
                    
                





        




