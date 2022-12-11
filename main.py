import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import torch
from torchvision.io import read_image

from environment import ApplicationProperties
from helper import Helper
from segmentation import ImageSegmentation
from transformation import InverseTransform
from yolo import DrivewayDetection

import multiprocess as mp
from itertools import repeat

def visualize_birdseye(masked_inv, da_inv, ll_inv):
    res = masked_inv.copy()
    # drivable area should be in green with alpha 0.5
    res[da_inv, :] = [0, 255, 0]
    # lane lines should be in red with alpha 0.5
    res[ll_inv, :] = [255, 0, 0]
    return res

np.set_printoptions( precision=3, suppress=True )
pool = mp.Pool(mp.cpu_count())

applicationProperties = ApplicationProperties( propertiesFilePath="application.yml" )
applicationProperties.initializeProperties()

lane_images_path = applicationProperties.get_property_value( "input.lane_images_path" )
rgb_images_path = applicationProperties.get_property_value( "input.scene_images_path" )

calib_path = applicationProperties.get_property_value( "calibration.file_path" )

K = Helper.parse_calibrations( calib_path )

rgb_images_dir = sorted( glob.glob( os.path.join( rgb_images_path, '*.png' ) ) )
# lane_images_dir = sorted( glob.glob( os.path.join( lane_images_path, '*.png' ) ) )
rgb_img_list = [ read_image( img_path ) for img_path in rgb_images_dir ]
original_image_shape = rgb_img_list[0].shape

# Preparing Template Car Image
template_car_image = cv2.imread( applicationProperties.get_property_value( "input.template_car_image_path" ) )
template_car_image = cv2.resize( template_car_image, ( 80, int( 80 * template_car_image.shape[0] / template_car_image.shape[1] ) ), interpolation=cv2.INTER_AREA)

img_output, masked_scene_img = ImageSegmentation.performImageSegmentation( 
    pool,
    rgb_img_list,
    applicationProperties.get_property_value( "segmentation.min_threshold" ),
    applicationProperties.get_property_value( "segmentation.alpha" )
)

birds_eye_view_of_scene = InverseTransform.performInversePerspectiveTransformation(
    masked_scene_img,
    K,
    img_output,
    applicationProperties.get_property_value( "inverse_transformation.min_threshold" ),
    template_car_image,
    applicationProperties.get_property_value( "inverse_transformation.min_vertical_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.max_vertical_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.min_horizontal_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.max_horizontal_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.image_dimensions" )
)

# YOLOP Implementation
yolo_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
da_seg, ll_seg = DrivewayDetection.performDrivewayDetection( 
    yolo_model, 
    rgb_img_list[-1],
    applicationProperties.get_property_value( "yolo.threshold" ),
    applicationProperties.get_property_value( "yolo.kernel_size" ),
    (
        applicationProperties.get_property_value( "yolo.out_image_height" ),
        applicationProperties.get_property_value( "yolo.out_image_width" )
    )
)

da_ll_3 = np.zeros( ( da_seg.shape[0], da_seg.shape[1], 3 ) )
da_ll_3[:,:,0] = da_seg
da_ll_3[:,:,1] = ll_seg

birds_eye_view_of_lanes_and_driveways = InverseTransform.inverse_perspective_transform(
    da_ll_3,
    K,
    applicationProperties.get_property_value( "inverse_transformation.min_vertical_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.max_vertical_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.min_horizontal_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.max_horizontal_clip" ),
    applicationProperties.get_property_value( "inverse_transformation.image_dimensions" )
)

da_inv = birds_eye_view_of_lanes_and_driveways[ :,:,0 ].astype( bool )
ll_inv = birds_eye_view_of_lanes_and_driveways[ :,:,1 ].astype( bool )

res = visualize_birdseye(birds_eye_view_of_scene, da_inv, ll_inv)
res = res[:, :, [2, 1, 0]]

# cv2.imwrite("output.png", birds_eye_view_of_scene)
cv2.imwrite("output.png", res)