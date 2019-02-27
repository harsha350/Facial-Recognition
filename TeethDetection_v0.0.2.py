
# coding: utf-8

# In[2]:


# import the necessary packages
from imutils import face_utils
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import argparse
import matplotlib
matplotlib.use("agg")
import imutils
import dlib
import cv2
from roipoly import roipoly 
import matplotlib.path as mplPath
import time
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


import sys, getopt, re

def main(argv):
    inputfile = ''
    deltaB=''
    output=''
    crop=''
    pcrop=''
    try:
        opts, args = getopt.getopt(argv,"i:d:o:c:p:",["input","delta","output","crop","postcrop"])
        #print(opts,args)
    except getopt.GetoptError:
        print('usage: TeethDetection_for_cropped.py -i <inputfile> -d <delta B value> -o <outputfile> -c <pre cropped path> -p <post cropped path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i","--input"):
            inputfile = arg
            #print(arg)
        elif opt in ("-d","--delta"):
            deltaB = int(arg)
            #print(arg)

        elif opt in ("-o","--output"):
            output = arg
            #print(arg)
        elif opt in ("-c","--crop"):
            crop = arg
            #print(arg)
        elif opt in ("-p","--postcrop"):
            pcrop = arg
            #print(arg)
        
            
    if len(inputfile)==0:
        print('usage: TeethDetection_for_cropped.py -i <inputfile> -d <delta B value> -o <outputfile> -c <pre cropped path> -p <post cropped path>')
        sys.exit(2)
    #print('Input file is ', inputfile)
    #print('Delta B is ', str(deltaB))
    #print('Output file is in ', output)
    return {'input':inputfile,'deltaB':deltaB,'output':output,'crop':crop, 'pcrop':pcrop}







def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical inner mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0],mouth[4])
 
    # compute the mouth aspect ratio
    mouth_ratio = (A +B+C) / (3.0 * D)
 
    # return the eye aspect ratio
    return mouth_ratio

def deltaLab(image,x_updated_pos,y_updated_pos,delta):
    
    # Converting Image bgr to lab
    image_lab=cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_up=[]
    # Take the x pos and y pos
    if (len(x_updated_pos)!=0 and len(y_updated_pos)!=0):
        for i in range(3):
            image_updated=image_lab[x_updated_pos,y_updated_pos,i]
            image_updated2=np.double(image_updated)+delta[i]
            image_updated2[np.where(image_updated2<120)]=120
            image_lab[x_updated_pos,y_updated_pos,i]=np.uint8(image_updated2)
            image_up=cv2.cvtColor(image_lab,cv2.COLOR_Lab2BGR)   #cv2.imshow("Result"+str(i),image_up)
    return image_up

    
## Sparkle image function 

def sparkle_image(image,point,image_sparkle):
    (B,A,_)=image.shape
    
    width_size=int(A*0.12)
    
    image_sparkle_1=imutils.resize(image_sparkle, width=width_size)
    (b,a,_)=image_sparkle_1.shape
    (x,y)=point
    
    up=int(y-b/2)
    down=B-up-b
    left=int(x-a/2)
    right=A-left-a
    image_sparkle_new=cv2.copyMakeBorder(image_sparkle_1,up,down, left, right,cv2.BORDER_CONSTANT,value=[0,0,0])
    return image_sparkle_new
    

def sparkle_image2(image,point1,point2,image_sparkle):
    (B,A,_)=image.shape
    
    width_size=int(A*0.12)
    
    image_sparkle_1=imutils.resize(image_sparkle, width=width_size)
    (b,a,_)=image_sparkle_1.shape
    (x,y)=point1
    up=int(y-b/2)
    down=B-up-b
    left=int(x-a/2)
    right=A-left-a
    image_sparkle_new1=cv2.copyMakeBorder(image_sparkle_1,up,down, left, right,cv2.BORDER_CONSTANT,value=[0,0,0])
    
    (x,y)=point2
    up=int(y-b/2)
    down=B-up-b
    left=int(x-a/2)
    right=A-left-a
    image_sparkle_new2=cv2.copyMakeBorder(image_sparkle_1,up,down, left, right,cv2.BORDER_CONSTANT,value=[0,0,0])
    image_sparkle_new=image_sparkle_new1+image_sparkle_new2
    return image_sparkle_new
    
#image_updated2=np.double(image_updated)+i*6
#image_updated2[np.where(image_updated2>255)]=255
#image_up=image.copy()
#image_up[x_updated_pos,y_updated_pos,:]=image_updated2
#cv2.imshow("Result"+str(i),image_up)
#cv2.imwrite("Output/image"+str(i)+".jpg",image_up)





# In[3]:


def normalized(down):

        norm=np.zeros((600,800,3),np.float32)
        norm_rgb=np.zeros((600,800,3),np.uint8)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        sum=b+g+r

        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0

        norm_rgb=cv2.convertScaleAbs(norm)
        return norm_rgb


# ## Teeth Detection

# In[4]:


#print(sys.argv[1:])
#argv=['i','face5.jpg','','']

inputArg=main(sys.argv[1:])
logFile=inputArg["output"].split(".")[0]+"_log.txt"
#f=open(logFile,"w+")

def main_method():
    tic=time.time()
    #inputArg={'input':"face5.jpg",'deltaB':"-5",'output':"face5"}
    #image_url="face5.jpg"
    image_url=inputArg['input']
    local=False
    #f.write("Started -"+ str(time.time())+"\n")


    
    image = cv2.imread(image_url)

    #f.write("Image Read -"+ str(time.time())+"\n")
    image_type="Lab"
    #image = imutils.resize(image, width=640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    predictor_url=""




    detector = dlib.get_frontal_face_detector()
    if(local):
        predictor_url="shape_predictor_68_face_landmarks.dat"
    else:
        predictor_url ="D:/home/site/wwwroot/crest/imgproc/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_url)

    rects = detector(gray, 1)
    
    #f.write("64 points detected -"+ str(time.time())+"\n")

    shape=[]
    for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            l=[67,66,65,64,63,62,61,60]
            
            #shape=[shape[i] for i in l]
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box


            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            #for (x, y) in shape:
                    #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks

    #pts=shape[[67,66,65,64,63,62,61,60]]
    pt1=np.int0(shape[[61,60,67]].mean(axis=0))
    pt2=np.int0(shape[[63,64,65]].mean(axis=0))
    pt3=np.int0(shape[[63,64,55]].mean(axis=0))
    pt4=np.int0(shape[[61,60,59]].mean(axis=0))

    mouth_thres=0.17

    pts_inner_mouth=shape[[60,61,62,63,64,65,66,67]]
    #print(mouth_aspect_ratio(pts_inner_mouth))
    if(mouth_aspect_ratio(pts_inner_mouth)<mouth_thres or mouth_aspect_ratio(pts_inner_mouth)>0.5):
        raise Exception('NOTEETH')
    #f.write("No Teeth Detection completed -"+ str(time.time())+"\n")


    #pts=shape[48:68]
    pt_1=shape[[67,66,65,64,63,62,61,60]]
    pt_2=shape[[58,57,56,54,53,51,49,48]]
    pt_3=shape[[58,57,55,54,53,51,50,48]]
    pts=np.int0((2.2*pt_1+0.4*pt_2+0.4*pt_3)/3)

    hull = cv2.convexHull(pts)
    overlay=image.copy()
    color=(19, 199, 109)

    vert=[(i[0][0],i[0][1]) for i in hull]
    vert.append(vert[0])



    def getMask(poly_verts, currentImage):
        ny, nx = np.shape(currentImage)

            # Create vertex coordinates for each grid cell...
            # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        ROIpath = mplPath.Path(poly_verts)
        grid = ROIpath.contains_points(points).reshape((ny,nx))
        return grid

    currentImage=image[:,:,1]
    ny, nx = np.shape(currentImage)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    ROIpath = mplPath.Path(vert)
    image_bool=getMask(vert,image[:,:,1])
    image_bool=image_bool*255
    (x_pos,y_pos)=np.where(image_bool==255)

    image_type=image.copy()
    #image_type=cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    z_img=image_type[x_pos,y_pos,:]



    # In[ ]:


    num=2
    method="kMean"
    if(method=="kMean"):
        model = KMeans(n_clusters=num, random_state=0).fit(z_img)
    elif(method=="gmm"):
        model=GaussianMixture(n_components=num).fit(z_img)
    elif(method=="agc"):
        dendrogram = sch.dendrogram(sch.linkage(z_img, method='ward'))
        model = AgglomerativeClustering(n_clusters=num, affinity = 'euclidean', linkage = 'ward').fit(z_img)


    #f.write("Clustering completed -"+ str(time.time())+"\n")
        

    label=np.array(model.predict(z_img))

    mean_=[]
    for i in range(num):
            #image_type=cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            #z_img=image_type[x_pos,y_pos,:]
            sum_=[t[0] for t in z_img[np.where(label==i)]]
            mean_.append(sum(sum_)/len(sum_))
    #print(np.abs(mean_[0]-mean_[1]))

    if(np.abs(mean_[0]-mean_[1])<50):
        raise Exception('NOTEETH')
    label_pos=mean_.index(max(mean_))
            
            
    teeth_label_pos=np.where(label==label_pos)
    x_updated_pos=x_pos[teeth_label_pos]
    y_updated_pos=y_pos[teeth_label_pos]
    image_bool_1=image_bool.copy()*0
    image_bool_1[x_updated_pos,y_updated_pos]=255


    image_updated=image[x_updated_pos,y_updated_pos,:]

    image_updated=image[x_updated_pos,y_updated_pos,:]

    deltaB=int(inputArg['deltaB'])
    output=inputArg['output']
    image_up=image.copy()
    #output="face5"

    image_up=deltaLab(image,x_updated_pos,y_updated_pos,[0,0,deltaB])

    #f.write("Delta B completed -"+ str(time.time())+"\n")

    #cv2.imwrite("Output/"+output,image_up)


    # ## Adding Sparkles

    # In[ ]:


    #Reading sparkle image
    if(local):
        sparkle_url="Sparkles/sp08.jpg"
    else:
        sparkle_url ="D:/home/site/wwwroot/crest/imgproc/sp08.jpg"

    image_sparkle=cv2.imread(sparkle_url)
    #image_sparkle=cv2.imread()

    x=imutils.resize(image_sparkle, width=50)

    #Generating image for sparkles at given points
    im_s=sparkle_image(image,pt3,image_sparkle)
    #f.write("Sparkle image completed -"+ str(time.time())+"\n")

    #Adding sparkle image and image 
    im_comb=np.where(image_up>im_s,image_up,im_s)
    #f.write("Output Saved -"+ str(time.time())+"\n")

    #Save
    cv2.imwrite(output,im_comb)

    crop_path=inputArg['crop']
    #print(crop_path)
    pcrop_path=inputArg['pcrop']
    #print(pcrop_path)
    #Cropped Image

    x_=pts[:,0]
    y_=pts[:,1]
    x_s=[x_.min()-5,x_.max()+5]
    y_s=[y_.min()-5,y_.max()+5]


    image_cropped_pre=image[y_s[0]:y_s[1],x_s[0]:x_s[1],:]
    image_cropped_post=image_up[y_s[0]:y_s[1],x_s[0]:x_s[1],:]

    cv2.imwrite(crop_path,image_cropped_pre)
    cv2.imwrite(pcrop_path,image_cropped_post)
    #f.write("Cropped and Post Cropped Completed-"+ str(time.time())+"\n")

    toc=time.time()
    print("SUCCESS")
    #return f

try:
    main_method()
    #f.write("Completed in -"+ str(time.time())+"\n")
except:
    print("UNIDENT")
    #f.write("UNIDENT in -"+ str(time.time())+"\n")
#f.close()
