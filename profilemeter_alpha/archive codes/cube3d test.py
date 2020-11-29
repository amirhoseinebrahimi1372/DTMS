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
if __name__ == '__main__':


        ret = np.load(path_cal + '/ret.npy')
        mtx = np.load(path_cal + '/K.npy')
        dist = np.load(path_cal + '/dist.npy')
        rvecs = np.load(path_cal + '/rvecs.npy')
        tvecs = np.load(path_cal + '/tvecs.npy')

        cam = cameraGraber(0)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        
        while True:
            img = next( cam.grabber(rotate = None ) )


            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img, deg = draw1(img,corners2,imgpts)
            cv2.imshow('img',cv2.resize(img, None, fx=0.5, fy=0.5))

            ideal = [0,90,90]
            ideal = np.copysign( ideal, deg)
            error = ideal - deg
            #dst = wrapp(img,gamma = error[0], phi = error[2], theta = error[1])
            dst = wrapp(img,gamma = error[0])
            print(error)
            print(deg)
            cv2.imshow('dst',cv2.resize(dst, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(50) & 0xff
            if key == 27:
                break
            
            gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img, deg = draw1(dst,corners2,imgpts)

                print('aaaaaaaaa', deg)
                print()
            if key == 27:
                break
            


                    
                    
                





        




