import cv2
import numpy as np
import numba
from numba import cuda
from pypylon import pylon
import time
import os
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




#________________________________________________________________________________________________________________________________________
#                                                             Main 
#________________________________________________________________________________________________________________________________________

def take_img(idx, imgs_path):
    name_idx = 49
    cam = cameraGraber(idx)
    while True:
        img = next( cam.grabber(rotate = None ) )
        cv2.imshow( 'img_' + str(name_idx), cv2.resize( img, None,fx=0.5, fy=0.5))
        key = cv2.waitKey(30)
        if key == ord('s'):
            print('saved')
            cv2.imwrite( os.path.join( imgs_path, 'cal_' +  str(name_idx )+ '.jpg'), img )
            cv2.destroyAllWindows()
            name_idx +=1
        if key == 27:
            break
                                       
        


class CameraCalibration():

    def __init__(self, path ):
        self.path_cal = path
        self.ret = []
        self.K = []
        self.dist = []
        self.rvecs = []
        self.tvecs =[]




    def take_img(self, imgs_path, cam_idx ):
        name_idx = 0
        cam = cameraGraber(cam_idx)
        while True:
            img = next( cam.grabber(rotate = None ) )
            cv2.imshow( 'img_' + str(name_idx), cv2.resize( img, None,fx=0.5, fy=0.5))
            key = cv2.waitKey(30)
            if key == ord('s'):
                print('saved')
                cv2.imwrite( os.path.join( imgs_path, 'cal_' +  str(name_idx )+ '.jpg'), img )
                cv2.destroyAllWindows()
                name_idx +=1
            if key == 27:
                break



        
    def calc_calibration(self, imgs_path, chessboard_size = (6,6) ):
        
        obj_points = [] #3D points in real world space 
        img_points = [] #3D points in image plane
        #Prepare grid and points to display
        objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    
        

    
        #Iterate over images to find intrinsic matrix
        for image_file in os.listdir(imgs_path):
            #Load image
            image = cv2.imread( os.path.join(imgs_path, image_file))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Image loaded, Analizying...")
            #find chessboard corners
            ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            
            if ret == True:
                print("Chessboard detected!")
                image = cv2.drawChessboardCorners(image, chessboard_size, corners,ret)
                #define criteria for subpixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                #refine corner location (to subpixel accuracy) based on criteria.
                cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
                obj_points.append(objp)
                img_points.append(corners)


            cv2.imshow('image', cv2.resize(image, None, fx=0.5, fy=0.5))
            cv2.waitKey(50)
        print('Analyz End')
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)
        #Save parameters into numpy file
        self.ret = np.save(self.path_cal + '/ret', ret )
        self.K = np.save(self.path_cal + '/K', K)
        self.dist = np.save(self.path_cal + '/dist', dist)
        self.rvecs = np.save(self.path_cal + '/rvecs', rvecs)
        self.tvecs = np.save(self.path_cal + '/tvecs', tvecs)

        self.ret =  ret
        self.K = K
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs =  tvecs

        tot_error = 0
        for i in range(len(obj_points)):
            imgpoints_cal, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i],imgpoints_cal, cv2.NORM_L2)/len(imgpoints_cal)
            tot_error += error

        print( 'error: ' , tot_error )
        return tot_error





    def load(self):
        self.ret = np.load(self.path_cal + '/ret.npy')
        self.K = np.load(self.path_cal + '/K.npy')
        self.dist = np.load(self.path_cal + '/dist.npy')
        self.rvecs = np.load(self.path_cal + '/rvecs.npy')
        self.tvecs = np.load(self.path_cal + '/tvecs.npy')




        
    def undistort( self, img ):
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K,self.dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, self.K, self.dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
 
        
    
        


if __name__ == '__main__':
    camera_calibration = CameraCalibration('data/camera_calib')
    #camera_calibration.calc_calibration('data/cal_img', chessboard_size = (9,6))
    camera_calibration.load()

    while True:
        cam = cameraGraber(0)
        img = next(cam.grabber())
        dst = camera_calibration.undistort(img)
        cv2.imshow('orginal img',cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.imshow('unidstor img',cv2.resize(dst, None, fx=0.5, fy=0.5))
        cv2.waitKey(10)

    '''

    if state == 0:
        take_img(0,imgs_path)

    elif state == 1:
        
        #============================================
        # Camera calibration
        #============================================
        #Define size of chessboard target.
        chessboard_size = (9,6)
        obj_points = [] #3D points in real world space 
        img_points = [] #3D points in image plane
        #Prepare grid and points to display
        objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

        

    
        #Iterate over images to find intrinsic matrix
        for image_file in os.listdir(imgs_path):
            #Load image
            image = cv2.imread( os.path.join(imgs_path, image_file))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Image loaded, Analizying...")
            #find chessboard corners
            ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            
            if ret == True:
                print("Chessboard detected!")
                image = cv2.drawChessboardCorners(image, chessboard_size, corners,ret)
                #define criteria for subpixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                #refine corner location (to subpixel accuracy) based on criteria.
                cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
                obj_points.append(objp)
                img_points.append(corners)


            cv2.imshow('image', cv2.resize(image, None, fx=0.5, fy=0.5))
            cv2.waitKey(200)
        print('Analyz End')
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)
        #Save parameters into numpy file
        np.save(path_cal + '/ret', ret)
        np.save(path_cal + '/K', K)
        np.save(path_cal + '/dist', dist)
        np.save(path_cal + '/rvecs', rvecs)
        np.save(path_cal + '/tvecs', tvecs)
        state = 3

        tot_error = 0
        for i in range(len(obj_points)):
            imgpoints_cal, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i],imgpoints_cal, cv2.NORM_L2)/len(imgpoints_cal)
            tot_error += error

        print( 'error: ' , tot_error )



    elif state == 2:
        ret = np.load(path_cal + '/ret.npy')
        K = np.load(path_cal + '/K.npy')
        dist = np.load(path_cal + '/dist.npy')
        rvecs = np.load(path_cal + '/rvecs.npy')
        tvecs = np.load(path_cal + '/tvecs.npy')

        cam = cameraGraber(0)
        while True:
            img = next( cam.grabber(rotate = None ) )
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

            dst = cv2.undistort(img, K, dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            cv2.imshow( 'img', cv2.resize( img, None,fx=0.5, fy=0.5))
            cv2.imshow( 'cal', cv2.resize( dst, None,fx=0.5, fy=0.5))
            cv2.waitKey(100)
    '''

                    
                    
                





        




