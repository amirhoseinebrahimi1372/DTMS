from pypylon import pylon
import cv2

'''
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
#cameras = pylon.InstantCameraArray(len(devices))
camera = pylon.InstantCamera()
camera.Attach(tlFactory.CreateDevice(devices[1]))
camera.PixelFormat = pylon.PixelType_BGR10V1packed
        
#camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
    camera.Width.SetValue(new_width)

numberOfImagesToGrab = 100
camera.StartGrabbing()

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data.
        print("SizeX: ", grabResult.Width)
        print("SizeY: ", grabResult.Height)
        img = grabResult.Array
        cv2.imshow('img', cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.waitKey(30)
        print("Gray value of first pixel: ", img[0, 0])

    grabResult.Release()
camera.Close()
'''


'''
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
#cameras = pylon.InstantCameraArray(len(devices))
camera = pylon.InstantCamera()
camera.Attach(tlFactory.CreateDevice(devices[1]))
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing()
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        #if rotate != None:
        #    img = cv2.rotate( img, rotate )
        cv2.imshow('img', cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.waitKey(30)
    grabResult.Release()


'''


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
        #pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        #------------------------------------------------------------------------------------
        
        '''
        self.camera.GainAuto.SetValue('Off')
        self.camera.GainSelector.SetValue('All')
        self.camera.GainRaw.SetValue(100)
        #self.camera.GammaSelector.SetValue('User')
        #cameras['ul'].camera.BlackLevelSelector.SetValue('All')

        #self.camera.LightSourceSelector.SetValue('Daylight 6500 Kelvin')
        self.camera.BalanceWhiteAuto.SetValue('Off')
        self.camera.ColorAdjustmentEnable.SetValue(False)
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.ExposureTimeRaw.SetValue(3000)
        self.camera.ExposureTimeAbs.SetValue(3000)
        '''
        
        
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
        
        
        


    def grabber( self, rotate=None ):

        self.camera.StartGrabbing()
        
        
        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                

                #if rotate != None:
                #    img = cv2.rotate( img, rotate )
                yield img
            grabResult.Release()


c = cameraGraber(0)

while True:
    img =next( c.grabber())
    cv2.imshow('img',img)
    cv2.waitKey(30)
