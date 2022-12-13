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

import time

def visualize_birdseye(masked_inv, da_inv, ll_inv):
    res = masked_inv.copy()
    # drivable area should be in green with alpha 0.5
    res[da_inv, :] = [0, 255, 0]
    # lane lines should be in red with alpha 0.5
    res[ll_inv, :] = [255, 0, 0]
    return res

if __name__ == "__main__":

    np.set_printoptions( precision=3, suppress=True )
    pool = mp.Pool(mp.cpu_count())

    print("reading applications file")
    applicationProperties = ApplicationProperties( propertiesFilePath="application.yml" )
    applicationProperties.initializeProperties()

    lane_images_path = applicationProperties.get_property_value( "input.lane_images_path" )
    rgb_images_path = applicationProperties.get_property_value( "input.scene_images_path" )
    output_images_path = applicationProperties.get_property_value( "output.image_directory" )

    calib_path = applicationProperties.get_property_value( "calibration.file_path" )

    K = Helper.parse_calibrations( calib_path )

    rgb_images_dir = sorted( glob.glob( os.path.join( rgb_images_path, '*.png' ) ) )
    # lane_images_dir = sorted( glob.glob( os.path.join( lane_images_path, '*.png' ) ) )
    rgb_img_list = [ read_image( img_path ) for img_path in rgb_images_dir ]
    original_image_shape = rgb_img_list[0].shape

    # Preparing Template Car Image
    template_car_image = cv2.imread( applicationProperties.get_property_value( "input.template_car_image_path" ) )
    template_car_image = cv2.resize( template_car_image, ( 80, int( 80 * template_car_image.shape[0] / template_car_image.shape[1] ) ), interpolation=cv2.INTER_AREA)

    print("performing image segmentation")
    time1 = time.time()
    img_outputs, masked_scene_imgs = ImageSegmentation.performImageSegmentation( 
        pool,
        rgb_img_list,
        applicationProperties.get_property_value( "segmentation.min_threshold" ),
        applicationProperties.get_property_value( "segmentation.alpha" )
    )
    print("finished image segmentation in: ", time.time() - time1)

    print("performing perspective inverse transformation")
    birds_eye_view_of_scene = InverseTransform.performInversePerspectiveTransformation(
        pool,
        masked_scene_imgs,
        K,
        img_outputs,
        applicationProperties.get_property_value( "inverse_transformation.min_threshold" ),
        template_car_image,
        applicationProperties.get_property_value( "inverse_transformation.min_vertical_clip" ),
        applicationProperties.get_property_value( "inverse_transformation.max_vertical_clip" ),
        applicationProperties.get_property_value( "inverse_transformation.min_horizontal_clip" ),
        applicationProperties.get_property_value( "inverse_transformation.max_horizontal_clip" ),
        applicationProperties.get_property_value( "inverse_transformation.image_dimensions" )
    )

    # YOLOP Implementation
    print("performing yolo")
    yolo_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    yolo_model = yolo_model.to( 'mps' )
    da_ll_3s = DrivewayDetection.performDrivewayDetection( 
        pool,
        yolo_model, 
        rgb_img_list,
        applicationProperties.get_property_value( "yolo.threshold" ),
        applicationProperties.get_property_value( "yolo.kernel_size" ),
        (
            applicationProperties.get_property_value( "yolo.out_image_height" ),
            applicationProperties.get_property_value( "yolo.out_image_width" )
        )
    )

    print("performing inverse_perspective_transform for lanes")
    pool = mp.Pool(mp.cpu_count())
    birds_eye_view_of_lanes_and_driveways = pool.starmap(
        InverseTransform.inverse_perspective_transform_pool,
        zip(
            repeat(da_ll_3s),
            repeat(K),
            repeat(applicationProperties.get_property_value( "inverse_transformation.min_vertical_clip" )),
            repeat(applicationProperties.get_property_value( "inverse_transformation.max_vertical_clip" )),
            repeat(applicationProperties.get_property_value( "inverse_transformation.min_horizontal_clip" )),
            repeat(applicationProperties.get_property_value( "inverse_transformation.max_horizontal_clip" )),
            repeat(applicationProperties.get_property_value( "inverse_transformation.image_dimensions" )),
            range(len(da_ll_3s))
        )
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    print(rgb_img_list[0].shape)
    out = cv2.VideoWriter(f"{output_images_path}project_f.mp4", fourcc, 25, (1500, 1500), False )
    for i in range(len(birds_eye_view_of_lanes_and_driveways)):
        print("processing image: ", i)
        da_inv = birds_eye_view_of_lanes_and_driveways[i][ :,:,0 ].astype( bool )
        ll_inv = birds_eye_view_of_lanes_and_driveways[i][ :,:,1 ].astype( bool )

        res = visualize_birdseye(birds_eye_view_of_scene[i], da_inv, ll_inv)
        res = res[:, :, [2, 1, 0]]
        res = cv2.resize(res, (1500, 1500))
        out.write(res)

        # cv2.imwrite("output.png", birds_eye_view_of_scene)
        cv2.imwrite(f"{output_images_path}output_{i}.png", res)
    cv2.destroyAllWindows()
    out.release()
    print("released video")