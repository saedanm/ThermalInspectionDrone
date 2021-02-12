#Program utilizes custom trained DeepLabV3 model to segment solar PV from RGB-image
#HOW to install require packages:
#If computer has CUDA graphic card:
#>>pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#If no CUDA supported graphic card:
#>>pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#>>pip install opencv-python
#>>pip install matplotlib
#>>pip install tqdm
#>>pip install gpsphoto
#>>pip install haversine

#The program start from segmenting an RGB image, and then align the output mask to thermal image. 
#Finally use transorm mask with therma image to isolate only solar PV area

import torch
import numpy as np
import random
import torchvision
from torchvision import transforms, models
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

from GPSPhoto import gpsphoto
from haversine import haversine, Unit
#---------------------------------------------------------------
#Load custom module
from DeepHomography import DeepHomography
from DeepSolarPVSegmentation import DeepSolarPVSegmentation

from GeoTiff import GeoTiff


#---------------------------------------------------------------
def get_coordinate(jpg_filename):
    data = gpsphoto.getGPSData(jpg_filename)
    lat = data['Latitude']
    lng = data['Longitude']
    return (lat,lng)

def DrawHotspot(binarize_image, thermal_image):
    #Detect countour around each hotspot location
    contours, hierarchy = cv2.findContours(binarize_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        center = (int(x),int(y))
        radius = int(radius)
        if (radius<5):
            radius = 5

        #Draw circle on thermal image
        cv2.circle(thermal_image, center, radius, (0,0,255), 2)


print("Initialize deep homography")
deepHomography  = DeepHomography("ModelHomography4PV.pth")

print("Initialize deep segment")
deepSegment     = DeepSolarPVSegmentation("best-RGB_PVSegmentation_DeeplabV3.pth")

#Obtained all filename in Images folder
#Path for RGB images and theirs ground truth
input_image_filename = glob.glob("ERDI-22Dec-West\\RGB\\*.jpg")

print("Start to detect hotspot")
hotspot_temp = 40
cutoff_intensity = int(8.5*(hotspot_temp-20))
hotspot_data = []
for i in tqdm(range(len(input_image_filename))):
    #Load input image
    input_filename = input_image_filename[i]

    #Extract RGB index file
    base_filename, _ = os.path.splitext(os.path.basename(input_filename))
    index = base_filename[len(base_filename)-3:len(base_filename)] 

    #Get corresponding thermal image
    thermal_filename = "ERDI-22Dec-West\\Thermal-Gray\\image_thermal." +  index + ".jpg"
    thermal_gt_filename = "ERDI-22Dec-West\\Thermal-GroundTruth\\image_thermal." +  index + ".jpg"

    #load with opencv by default is color image (3 bands)
    rgb_input_image = cv2.imread(input_filename)
    gt_image    = cv2.imread(thermal_gt_filename, cv2.IMREAD_GRAYSCALE)
    thr_input_image = cv2.imread(thermal_filename)

    #Resize images to 128x128 for determine homography
    dim = (128, 128)
    rgb128 = cv2.resize(rgb_input_image, dim)
    thr128 = cv2.resize(thr_input_image, dim)

    #Resize image to 160x120
    rgb_input_image = cv2.resize(rgb_input_image, (160,120))
    gt_image        = cv2.resize(gt_image, (160,120))

    #Determine homography matrix from images 128x128 pixels
    homographyM = deepHomography.getDeepHomography(rgb128, thr128)

    #Segment RGBrped image (160x120 pixel)
    thr_mask = deepSegment.SegmentPVImage(rgb_input_image)

    #Warp thr_mask with homographyM
    thr_mask = cv2.warpPerspective(thr_mask, homographyM, (160,120))

    #Mask every layer of thermal image to show only PV
    image_pv =np.zeros((120,160), dtype=np.uint8)
    image_pv = np.bitwise_and(thr_input_image[:,:,0], thr_mask)

    #Threshold PV image to identify hotspot
    _,hotspot_im = cv2.threshold(image_pv,cutoff_intensity,255,cv2.THRESH_BINARY)

    hotspot_pixel = np.sum(hotspot_im)
    if (hotspot_pixel>0):
        #Extract GPS coordinate
        lat, lng = get_coordinate(thermal_filename)
        data = (int(index), hotspot_pixel, lat, lng)
        hotspot_data.append(data)
        DrawHotspot(hotspot_im, thr_input_image)
        filename = index + ".png"
        cv2.imwrite(filename,thr_input_image)

#Save hot spot data
hotspot_file = open("hotspot-dat.txt", "w")
#hotspot_file.write("No\t Spots\t Latitude\t Longitude\n")
for i in range(len(hotspot_data)):
    print("%0.0f\t%0.0f\t%f\t%f"%(hotspot_data[i][0], hotspot_data[i][1]/255, hotspot_data[i][2], hotspot_data[i][3]))
    hotspot_file.write("%d\t%0.0f\t%f\t%f\n"%(hotspot_data[i][0],hotspot_data[i][1]/255, hotspot_data[i][2], hotspot_data[i][3]))


#Load map
map = GeoTiff(filename="ERDI_Map.tif")
#Load map image
map_img = cv2.imread("ERDI_Map.tif")

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.6
fontColor              = (0,0,255)
lineType               = 2
#Try to locate coordinate on map
for i in range(len(hotspot_data)):
    lat = hotspot_data[i][2]
    lon = hotspot_data[i][3]
    x,y = map.World2Pixel(lon,lat)
    point = (x,y)
    cv2.circle(map_img, point, radius=0, color=(0,0,255), thickness=5)

    #Write image text
    point = (x+6,y)
    img_text= str(int(hotspot_data[i][0])).zfill(3)
    cv2.putText(map_img,img_text,point,font,fontScale,fontColor,lineType)
cv2.imwrite("hotspot.png", map_img)

