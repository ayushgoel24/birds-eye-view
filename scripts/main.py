import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

from PIL import Image
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

from environment import ApplicationProperties
from helper import Helper

np.set_printoptions( precision=3, suppress=True )

applicationProperties = ApplicationProperties( propertiesFilePath="application.yml" )

lane_images_path = applicationProperties.get_property_value( "input.lane_images_path" )
rgb_images_path = applicationProperties.get_property_value( "input.rgb_imgs" )

calib_path = applicationProperties.get_property_value( "calibration.file_path" )

K = Helper.parse_calibrations( calib_path )

rgb_images_dir = sorted( glob.glob( os.path.join( rgb_images_path, '*.png' ) ) )
lane_images_dir = sorted( glob.glob( os.path.join( lane_images_path, '*.png' ) ) )
rgb_img_list = [ read_image( img_path ) for img_path in rgb_images_dir ]
original_image_shape = rgb_img_list[0].shape