from pypylon import pylon
import cv2
import numpy as np
#import Plotter
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from matplotlib import cm
import time
import open3d as o3d
import meshlabxml as mlx
import os



#_________________________________________________________________________________________________________________
#                                               Mouse Event function
#_________________________________________________________________________________________________________________
mouse_x, mouse_y = -1,-1
mouse_event = None

def mouse(event,x,y,flags,param):
    global mouse_x, mouse_y, mouse_event
    mouse_x, mouse_y = x,y
    if event == cv2.EVENT_LBUTTONDOWN: 
        mouse_event = "LBUTTONDOWN"
        
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_event = "LBUTTONUP"

    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse_event = "RBUTTONDOWN"

    elif event == cv2.EVENT_RBUTTONUP:
        mouse_event = "RBUTTONUP"
        
    elif event == cv2.EVENT_MOUSEMOVE:
        mouse_event = "MOUSEMOVE"
    else :
        mouse_event = None

#-------- set Event mouse function on 'image' window
#cv2.namedWindow('frame')
#cv2.setMouseCallback('frame',mouse)

#_______________________________________________________________________________________________________________________________________________
#                                                       this Class Yield Frame of Camera
#_______________________________________________________________________________________________________________________________________________
class cameraGraber:
    def __init__(self):

        # conecting to the first available camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Grabing Continusely (video) with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        #-------------------------------------------------------------------------------------
        
        self.camera.Gamma.SetValue(2.0)
        self.camera.AutoGainUpperLimit.SetValue(0.)
        self.camera.AutoGainLowerLimit.SetValue(0.)
        self.camera.Gain.SetValue(0.)
        self.camera.BlackLevel.SetValue(3.8)
        self.camera.AutoExposureTimeLowerLimit.SetValue(10.)
        self.camera.AutoExposureTimeUpperLimit.SetValue(17055.)
        self.camera.ExposureTime.SetValue(17055.)
        self.camera.LightSourcePreset.SetIntValue(2)
        self.camera.SensorShutterMode.SetIntValue(0)
    
        self.camera.AutoExposureTimeUpperLimit.SetValue(10189.)
        self.camera.ExposureTime.SetValue(10189.)
        
        
        #-------------------------------------------------------------------------------------
        # converting to opencv bgr format
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


class laserDetector:

    @staticmethod
    def detcet( img ):
        blue,green,red = cv2.split(img)
        red = cv2.blur(red,(3,3))
        _,laser_mask = cv2.threshold(red, 100, 255, cv2.THRESH_BINARY )
        laser_mask = cv2.erode( laser_mask, (3,3), iterations=2 )
        #_,blue_not_mask = cv2.threshold(blue, 80, 255, cv2.THRESH_BINARY_INV )
        #_,green_not_mask = cv2.threshold(green,80, 255, cv2.THRESH_BINARY_INV )
        
        #laser_mask = cv2.erode( laser_mask, (3,3), iterations=1 )
        #laser_mask = cv2.dilate( laser_mask, (5,5), iterations=5)

        #blue_not_mask = cv2.erode( blue_not_mask, (3,3), iterations=10 )
        #green_not_mask = cv2.erode( green_not_mask, (3,3), iterations=10 )

        #laser_mask = cv2.bitwise_and( blue_not_mask, laser_mask )
        #laser_mask = cv2.bitwise_and( green_not_mask, laser_mask )
        return laser_mask




    @staticmethod
    def detcetCanny(img ):
        blue,green,red = cv2.split(img)
        laser_mask = cv2.Canny(red, 150, 200 )
        #laser_mask = cv2.dilate( laser_mask, (2,2), iterations=15 )

        _,red_mask = cv2.threshold(red, 70, 255, cv2.THRESH_BINARY )
        _,blue_not_mask = cv2.threshold(blue, 50, 255, cv2.THRESH_BINARY_INV )
        _,green_not_mask = cv2.threshold(green,50, 255, cv2.THRESH_BINARY_INV )

        red_mask = cv2.dilate( red_mask, (3,3), iterations=10 )
        blue_not_mask = cv2.erode( blue_not_mask, (3,3), iterations=10 )
        green_not_mask = cv2.erode( green_not_mask, (3,3), iterations=10 )

        laser_mask = cv2.bitwise_and( blue_not_mask, laser_mask )
        laser_mask = cv2.bitwise_and( green_not_mask, laser_mask )
        laser_mask = cv2.bitwise_and( red_mask, laser_mask )
        return laser_mask


    def detcet2(img, tresh=50 ):
        blue,green,red = cv2.split(img)
        sub = red - ( blue + green )/2
        sub = cv2.blur( sub, (3,3))
        h,w = sub.shape
        res = np.zeros(sub.shape, dtype = np.uint8 )
        ys = np.argmax( sub, axis=0 )
        xs = np.arange( w )
        res[ys,xs] = 255
        idx_tresh_not = (  sub < tresh )
        res[idx_tresh_not] = 0
        return res
            
    def detcet3(img, thresh=50 ):
        blue,green,red = cv2.split(img)
        sub = red - ( blue + green )/2
        sub = cv2.blur( sub, (3,3))
        _,msk = cv2.threshold( sub, thresh, 255, cv2.THRESH_BINARY )
        return msk
            
        
    @staticmethod
    def out_window( img, mask , output_size=None ):
        
        LINE = 10
        if output_size == None:
            img = cv2.resize(img, None, fx=0.25, fy=0.25)
            mask = cv2.resize(mask, None, fx=0.25, fy=0.25)

        h,w,_ = img.shape
        result = np.zeros((h,w*2 + LINE ,3), dtype = np.uint8 )
        result[:,:w] =  img
        result[:,w:w+LINE, 0] =  255
        result[:,w+LINE:,2] =  mask

        return result

    @staticmethod
    def filter( mask, iteratons=1 ):
        res = np.copy( mask )
        res = cv2.morphologyEx( res, cv2.MORPH_CLOSE, (5,5))
        for _ in range(iteratons):
            kernel = np.ones((1,5), np.uint8)
            res = cv2.dilate(res, kernel, iterations=1)
            res = cv2.erode(res, kernel, iterations=1)

            kernel = np.ones((5,1), np.uint8)
            res = cv2.dilate(res, kernel, iterations=1)
            res = cv2.erode(res, kernel, iterations=1)

            kernel = np.array([[1,0,0],
                               [0,1,0],
                               [0,0,1]], np.uint8)
            res = cv2.dilate(res, kernel, iterations=2)
            res_laser = cv2.erode(res, kernel, iterations=2)

            kernel = np.array([[0,0,1],
                               [0,1,0],
                               [1,0,0]], np.uint8)
            res = cv2.dilate(res, kernel, iterations=2)
            res = cv2.erode(res, kernel, iterations=2)
        
        return res


    def filter2( mask, iteratons=1 ):
        res = np.copy( mask )
        for _ in range(iteratons):
            kernel = np.ones((1,5), np.uint8)
            res = cv2.dilate(res, kernel, iterations=1)
            res = cv2.erode(res, kernel, iterations=1)
        return res
        

    @staticmethod
    def perspective( pointsm, angle):

        pts = np.copy( points )
        pts[:,1] = pts[:,1] / np.sin( angle/180 * np.pi )
        return pts

    @staticmethod        
    def extract_points( mask, point_num=1000 ):
        points = []
        _,cnts,_ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        for cnt in cnts:
            pts = cnt.reshape(-1,2)
            points.extend(pts)
        points.sort( key = lambda x:x[0] )
        points = np.array(points)
        points[:,1] = mask.shape[0] - points[:,1]
        return np.array(points)


    @staticmethod        
    def extract__mean_points( mask, point_num=1000 ):
        h,w = mask.shape
        xs=[]
        ys=[]
        for i in range( 0,w ):
            col = mask[:,i]
            all_y = np.argwhere(col)
            if len( all_y ) > 0:
                x = i
                y = h - np.mean(all_y) -1
                ys.append(y)
                xs.append(x)

        xs = np.expand_dims(xs,axis=-1)
        ys = np.expand_dims(ys,axis=-1)
        points = np.hstack((xs,ys))
        
        return points

    def mean_points( points ):
        points = np.array( points )
        xs = points[:,0]
        ys = points[:,1] 
        ys_sum_per_xs = np.bincount( xs, ys )
        ys_count_per_xs = np.bincount( xs, np.ones(ys.shape, dtype = np.int16 ))
        exist_points = ys_count_per_xs > 0
        mean_y = ys_sum_per_xs[exist_points] / ys_count_per_xs[exist_points]
        xs = np.argwhere(exist_points)
        mean_y = np.expand_dims( mean_y, axis=-1 )
        mean_points = np.hstack(( xs, mean_y )).astype(np.int32)
        return mean_points


    @staticmethod
    def extract_up_points( mask ):
        h,w = mask.shape
        xs=[]
        ys=[]
        for i in range( 0,w ):
            col = mask[:,i]
            all_y = np.argwhere(col)
            if len( all_y ) > 0:
                x = i
                y = h - np.min(all_y) -1
                ys.append(y)
                xs.append(x)

        xs = np.expand_dims(xs,axis=-1)
        ys = np.expand_dims(ys,axis=-1)
        points = np.hstack((xs,ys))
        return points
            

    @staticmethod
    def fill_zero( points ):
        pts = np.copy( points)
        xmin = np.min( pts[:,0])
        xmax = np.max( pts[:,0])
        low_y = np.min( pts[:,1])
        if xmin!=0:
            pts = np.vstack( (pts, np.array([xmin-1, low_y ])) )
        #pts = np.vstack( (pts, np.array([xmax+1, low_y ])) )
        for x in range(int(xmin),int(xmax)):
            if x not in pts[:,0]:
                pts = np.vstack( (pts, np.array([x, low_y ])) )

        pts = list(pts)
        pts.sort( key = lambda x:x[0] )
        pts = np.array( pts )
        return pts

    @staticmethod
    def fill( points ):
        pts = np.copy( points)
        xmin = np.min( pts[:,0])
        xmax = np.max( pts[:,0])
        low_y = np.min( pts[:,1])
        for x in range(xmin,xmax):
            if x not in pts[:,0]:
                pts = np.vstack( (pts, np.array([x, low_y ])) )

        pts = list(pts)
        pts.sort( key = lambda x:x[0] )
        pts = np.array( pts )
        return pts

    @staticmethod
    def approxContour( points, eps=0.1 ):
        pts = points.astype(np.int32)
        pts = pts.reshape((-1,1,2))
        arc_len = cv2.arcLength( pts, False )
        epsilon  = arc_len * eps
        pts = cv2.approxPolyDP( pts, epsilon, False )
        pts = pts.reshape((-1,2))
        return pts

    def approx(points, min_dist = 5, search_w = 5):
        pts = []
        for idx in range(len(points) - search_w ):
            if len(pts)==0:
                pts.append(points[0])
                continue
            last_point = pts[-1]
            dist = 0
            for next_idx in range( search_w ):
                dist = dist + abs( points[ idx + next_idx ][1] - last_point[1] )

            if dist > min_dist:
                pts.append( np.copy(points[idx]))

            else :
                pts.append( np.array( [points[idx][0],pts[-1][1] ]) )

        return np.array( pts)


    def approx_slope(points, min_dist = 5, search_w = 5):
        pts = []
        for idx in range(2, len(points) - search_w ):
            if len(pts)==0:
                pts.append(points[0])
                pts.append(points[1])
                continue
            
            last_point  = pts[-1]
            y_slop = pts[-1][1] - pts[-2][1]
            predict_y = pts[-1][1]
            
            dist = 0
            for next_idx in range(1, search_w ):
                predict_y += y_slop
                dist = dist + abs( points[ idx + next_idx ][1] - predict_y)
                

            if dist > min_dist:
                pts.append( np.copy(points[idx]))

            else :
                pts.append( np.array( [points[idx][0],pts[-1][1] ]) )
                

        return np.array( pts)
        
    @staticmethod
    def plot( points, w,h ):
        xs = points[:,0]
        ys = points[:,1]
        plt.clf()
        plt.plot(xs, ys, 'bo', label='surface', markersize=1)
        plt.ylim(0,h)
        plt.xlim(0,w)
        plt.gca().set_aspect('equal')
        plt.title('Camera Surface')
        plt.legend()
        plt.pause(0.02)

    @staticmethod
    def imgPlot( points, w,h ):
        pts = points.astype(np.uint16)
        img = np.zeros((h,w), np.uint8 )
        xs = pts[:,0]
        ys = pts[:,1]
        img[ys,xs] = 255
        cv2.imshow("plot", img )

    @staticmethod
    def contourPlot( points, w,h ):
        pts = points.astype(np.int32)
        pts = pts.reshape((-1,1,2))
        img = np.zeros((h,w), np.uint8 )
        cv2.drawContours( img, [pts],0,255 )
        cv2.imshow("contourPlot", img )


    @staticmethod
    def scanSurface( points,X,Y,Z,step ):
        MIN_DIST = 5
        dist = MIN_DIST + 1
        if len(Y.shape)>1 and Y.shape[1]>1:
            sub = abs(Y[:, -1] - points[:,1])
            idx = np.argwhere( sub )
            distribution =  sub[idx] / sub[idx]
            dist =  sum(sub)/ (sum(distribution ) + 1e-5)

        if dist< MIN_DIST:
            print("skip")
            Z[:,-1] += step
            return X,Y,Z

        x = points[:,0]
        y = points[:,1]
        
        z = np.zeros(x.shape)
        x = np.reshape(x,(-1,1))
        y = np.reshape(y,(-1,1))
        z = np.reshape(z,(-1,1))
        print(x.shape)
        if len(X) == 0:
            X = x
        else :
            X = np.hstack( (X,x))


        if len(Y) == 0:
            Y = y

        else :
            Y = np.hstack( (Y,y))
            
        if len(Z) == 0:
            Z = z
        else :
            z = z + Z[-1][-1] + step
            Z = np.hstack( (Z,z))

        return X,Y,Z

    @staticmethod
    def scanCP( points, xyz, z_step = 1 , interpolation = False):
        #points(x,y,z)   _   #xyz = (x,y,z)
        if len(xyz)==0:
            z=0
        else :
            z = xyz[-1][1]


        if interpolation:
            for z_ in range(z_step ):
                z += 1
                pts = np.zeros( ( len(points),3 ))
                pts[:,0] = np.copy( points[:,0] )
                pts[:,1] = z
                pts[:,2] = np.copy( points[:,1] )
                xyz.extend( pts )
        else :
            z += z_step
            pts = np.zeros( ( len(points),3 ))
            pts[:,0] = np.copy( points[:,0] )
            pts[:,1] = z
            pts[:,2] = np.copy( points[:,1] )
            xyz.extend( pts )
        return xyz

    

    @staticmethod
    def preProcess(X,Y,Z ):
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        X = X.transpose()
        Y = Y.transpose()
        Z = Z.transpose()
        return X,Y,Z


    def point2Image( points, size, fit = True ):
        pts = points.astype(np.int16)
        img = np.zeros( size, dtype = np.uint8 )
        pts[:,1] = size[0] - pts[:,1] - 1
        img[pts[:,1], pts[:,0] ] = 255
        if fit:
            maxy = int( h - np.min(points[:,1] ) )
            miny = int( h - np.max(points[:,1] ) )
            img = img[miny-5:maxy+5,:]

        return img
        


#________________________________________________________________________________________________________________________________________
#                                                           Main
#________________________________________________________________________________________________________________________________________

def plot(X,Y,Z,sufrace_shape, sampeling= 1):
    h,w = sufrace_shape
    p_num,_ = X.shape
    smpl = np.arange(0,p_num, sampeling ).astype( np.int32 )
    X = X[smpl,:]
    Y = Y[smpl,:]
    Z = Z[smpl,:]
    zmin = np.min(Z)
    zmax = np.max(Z)
    norm = plt.Normalize(Y.min(), Y.max())
    colors = cm.viridis(norm(Y))
    rcount, ccount, _ = colors.shape
    fig = plt.figure(211)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(zmin - 5 ,5 + zmax)        
    ax.set_ylim(0,w)
    ax.set_zlim(0,h)
    ax.plot_surface(Z, X, Y,  rcount=rcount, ccount=ccount,facecolors=colors, shade=False)
    #surf.set_facecolor((0,0,0,0))
    plt.pause(0.2)
    #plt.show()



def mesh_poisson(pcd, smooth=0):
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.4, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    
    p_mesh_crop.compute_vertex_normals()
    p_mesh_crop.compute_triangle_normals()


    ARG = 10000
    mesh_lod = p_mesh_crop.simplify_quadric_decimation(ARG)
    p_mesh_crop.paint_uniform_color([0.7, 0.4, 0])
    #p_mesh_crop = p_mesh_crop.filter_smooth_laplacian(number_of_iterations=smooth)

    if smooth>0:
        print("smooth")
        mesh_lod = p_mesh_crop.filter_smooth_simple(number_of_iterations=smooth)
    else :
        print("Not smooth")
        mesh_lod = p_mesh_crop

    o3d.io.write_triangle_mesh( "mesh.stl", mesh_lod )
    o3d.visualization.draw_geometries_with_editing([mesh_lod])
    

#
#________________________________________________________________________________________________________________________________________
#                                                           Main
#________________________________________________________________________________________________________________________________________

plt.figure(num=100, figsize=(4, 6), dpi=100, facecolor='w', edgecolor='k')
xyz = []
Scan = True
camera = cameraGraber()
laser_detcetor = laserDetector()

counter = 0
masks = []
THRESH=80
MIN_AREA = 30
EPS = 0.05
mean=0;
BASE_LINE_SHIFT = 2

frame =  next(camera.grabber(rotate = cv2.ROTATE_90_CLOCKWISE))
#frame = frame[ :1300, 700:2000]

h,w,_ = frame.shape 
res_points = np.zeros((w,2))
points_counter = np.zeros(w)
res_points[:,0] = np.arange(0,w)
name = 0
z = 0
base_laser = []
frame_ = []
mask_ = np.ones( frame.shape[0:2], dtype = np.uint8 ) * 255
X = np.array([])
Y = np.array([])
Z = np.array([])
pre_mask = None
line = []

j=0
for frame in camera.grabber(rotate = cv2.ROTATE_90_CLOCKWISE):
    
    t= time.time()
    #frame = frame[ :1300, 700:2000]
    #cv2.imshow("frame", cv2.resize(frame, None, fx=0.25, fy =0.25  ))
    frame = cv2.blur( frame, (1,5))

    #cv2.line(frame, (0,840), (700,860), (0,255,0), thickness=40 )

        
    


    mask = laserDetector.detcet3( frame, 50 )
    result = laserDetector.out_window( frame, mask, output_size=None)
    cv2.imshow("result",result )
    key = cv2.waitKey(10)
    if key == 27:
        break

    if key == ord('s'):
        print("Scaning")
        Scan = True

    if j%10 == 0:
        cv2.imwrite("export/"+str(name)+".png", mask)
        name += 1
    j+=1
    if key == ord('e'):
        print("Export")
        cv2.imwrite("dataset/"+str(name)+".png", mask)
        name += 1    


    points = laserDetector.extract__mean_points(mask)

    
    if len( points ) < 10:
        continue
    
    
    
    
    mask1 = laserDetector.point2Image(points, mask.shape)
    laserDetector.perspective( points, 60 )
    points = laserDetector.approx( points )
    mask2 = laserDetector.point2Image(points, mask.shape)

    #corners = cv2.goodFeaturesToTrack( mask2.astype(np.uint8), 5, 0.98, 20 )
    #for corner in corners:
    #   cv2.circle(mask2, tuple(corner[0]), 5, (255))
    
    cv2.imshow("mask2",mask2)
    cv2.imshow("mask1",mask1)
    cv2.imwrite("mask2.png", mask2)

    #laserDetector.plot(points,w,h)
    Sampel = 5
    pts = []
    for i in range( len( points )):
        if i%Sampel == 0:
            pts.append( points[i] )

    points = np.array(pts)
    xyz = laserDetector.scanCP( points, xyz, z_step=10, interpolation = False)

    if pre_mask is not None:
        diff = cv2.absdiff( mask, pre_mask )
        _,diff = cv2.threshold( diff, 50, 1, cv2.THRESH_BINARY )
        #print("diff",np.sum( diff) )
        
    pre_mask = np.copy(mask)
    




    
    
    
    
    

#if Scan:
cv2.destroyAllWindows()
xyz = np.array( xyz )
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("sync.ply", pcd)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd])






meshlab_path = 'C:\\Program Files\\VCG\\\MeshLab\\meshlab.exe'
meshlabserver_path = 'C:\\Program Files\\VCG\\MeshLab'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']
orange_cube = mlx.FilterScript( file_in = "sync.ply",file_out = "sysc_norm.ply", ml_version='2020.07')
mlx.normals.point_sets( orange_cube, neighbors=200)
orange_cube.run_script()
time.sleep(1)
pcd = o3d.io.read_point_cloud("sysc_norm.ply")



def msh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    factor =  avg_dist * 3
    radius = factor * avg_dist   
    radi =  o3d.utility.DoubleVector([radius, radius * 2])
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radi)





    bpa_mesh.compute_vertex_normals()
    

    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    

    o3d.io.write_triangle_mesh("bpa_mesh.ply", bpa_mesh)




    #ARG = 10000
    #mesh_lod = bpa_mesh.simplify_quadric_decimation(ARG)
    #mesh_lod.compute_triangle_normals()
    #mesh_lod.compute_vertex_normals()
    if smooth>0:
        print("smooth")
        mesh_lod = p_mesh_crop.filter_smooth_simple(number_of_iterations=smooth)
    else :
        print("Not smooth")
        mesh_lod = p_mesh_crop
    o3d.visualization.draw_geometries([mesh_lod])

                
    #cv2.destroyAllWindows()
norm = []
for i in range( len( pcd.points )):
    norm.append( np.array([0,0,1]) )
norm = np.array( norm )
pcd.normals = o3d.utility.Vector3dVector( norm )


                
    #cv2.destroyAllWindows()



#o3d.visualization.draw_geometries([pcd])
mesh_poisson(pcd , smooth=0)
mesh_poisson(pcd , smooth=1)
#msh(ocd)
