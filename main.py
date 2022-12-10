import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

from torchvision.io import read_image

from environment import ApplicationProperties
from helper import Helper
from segmentation import ImageSegmentation
from transformation import InverseTransform

np.set_printoptions( precision=3, suppress=True )

applicationProperties = ApplicationProperties( propertiesFilePath="application.yml" )
applicationProperties.initializeProperties()

lane_images_path = applicationProperties.get_property_value( "input.lane_images_path" )
rgb_images_path = applicationProperties.get_property_value( "input.scene_images_path" )

calib_path = applicationProperties.get_property_value( "calibration.file_path" )

K = Helper.parse_calibrations( calib_path )

rgb_images_dir = sorted( glob.glob( os.path.join( rgb_images_path, '*.png' ) ) )
lane_images_dir = sorted( glob.glob( os.path.join( lane_images_path, '*.png' ) ) )
rgb_img_list = [ read_image( img_path ) for img_path in rgb_images_dir ]
original_image_shape = rgb_img_list[0].shape

# Preparing Template Car Image
template_car_image = cv2.imread( applicationProperties.get_property_value( "input.template_car_image_path" ) )
template_car_image = cv2.resize( template_car_image, ( 80, int( 80 * template_car_image.shape[0] / template_car_image.shape[1] ) ), interpolation=cv2.INTER_AREA)

img_output, masked_scene_img = ImageSegmentation.performImageSegmentation( 
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

cv2.imwrite("output.png", birds_eye_view_of_scene)