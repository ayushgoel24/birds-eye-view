import numpy as np
from helper import Helper
import matplotlib.pyplot as plt

class InverseTransform( object ):

    def __init__(self) -> None:
        pass

    @staticmethod
    def inverse_perspective_transform( image:np.ndarray, K:np.ndarray, min_vertical_clip:float, max_vertical_clip:float, min_horizontal_clip:float, max_horizontal_clip:float, image_dimensions:float ) -> np.ndarray:
	
        PM = Helper.get_PM(K)
        image_width = image.shape[1]
        image_height = image.shape[0]

        # q_cx, q_cy: coordinates of the image in camera frame
        q_cx, q_cy = np.meshgrid( np.arange(image_width), np.arange(image_height) )
        q_c = np.stack( ( q_cx.flatten(), q_cy.flatten(), np.ones_like( q_cx.flatten() ) ), axis=1 ).T

        # q_r: coordinates of the road plane
        q_r = np.linalg.inv( PM ) @ q_c
        q_r = q_r / q_r[2, :]
        q_rx = q_r[0, :]
        q_ry = q_r[1, :]
        # clip values of q_rx because vanishing points
        q_rx = np.clip( q_rx, min_vertical_clip, max_vertical_clip )
        # clip values of q_ry
        q_ry = np.clip( q_ry, min_horizontal_clip, max_horizontal_clip )

        virtual_image_width = image_dimensions
        virtual_image_height = image_dimensions

        virtual_image = np.zeros( ( virtual_image_height, virtual_image_width, 3 ), dtype=np.uint8 )
        q_rx, q_ry = np.meshgrid( np.linspace( np.min(q_rx), np.max(q_rx), virtual_image_width ),
                                np.linspace( np.min(q_ry), np.max(q_ry), virtual_image_height ) )
        q_r = np.stack( ( q_rx.flatten(), q_ry.flatten(), np.ones_like(q_rx.flatten()) ), axis=1 ).T
        q_c = PM @ q_r # points of the virtual image mapped back to original image
        q_c = q_c / q_c[2, :]
        q_cx = q_c[0, :]
        q_cy = q_c[1, :]
        q_cx = q_cx.reshape(virtual_image_height, virtual_image_width)
        q_cy = q_cy.reshape(virtual_image_height, virtual_image_width)
        virtual_image[:,:,0] = Helper.interp2(image[:,:,0], q_cx, q_cy).T
        virtual_image[:,:,1] = Helper.interp2(image[:,:,1], q_cx, q_cy).T
        virtual_image[:,:,2] = Helper.interp2(image[:,:,2], q_cx, q_cy).T
        # flip virtual image upside down
        virtual_image = virtual_image[::-1, :, :]
        # flip virtual image left to right
        virtual_image = virtual_image[:, ::-1, :]
        # swap red and blue channels
        virtual_image = virtual_image[:, :, [2, 1, 0]]
        
        return virtual_image

    @staticmethod
    def performInversePerspectiveTransformation( masked_img:np.ndarray, K:np.ndarray, img_output:dict, inverse_perspective_threshold:float, template_car:np.ndarray, min_vertical_clip:float, max_vertical_clip:float, min_horizontal_clip:float, max_horizontal_clip:float, image_dimensions:float ) -> np.ndarray:
        inv_masked_img = InverseTransform.inverse_perspective_transform( masked_img, K, min_vertical_clip, max_vertical_clip, min_horizontal_clip, max_horizontal_clip, image_dimensions )
        
        mask_bool = ( img_output['masks'] > inverse_perspective_threshold )[0][0].numpy()
        
        bool_lower_bound = max( np.where( mask_bool > 0 )[0] )
        bool_left_bound = min( np.where( mask_bool > 0 )[1] )
        bool_right_bound = max( np.where( mask_bool > 0 )[1] )
        bool_middle = ( bool_left_bound + bool_right_bound ) // 2

        bbox_bottom_center = np.array( [ bool_middle, bool_lower_bound ] )

        bbox_bottom_mapped = np.linalg.inv( Helper.get_PM(K) ) @ np.array( [ bbox_bottom_center[0], bbox_bottom_center[1], 1 ] )
        bbox_bottom_mapped = [ ( bbox_bottom_mapped[0] / bbox_bottom_mapped[2] ), ( bbox_bottom_mapped[1] / bbox_bottom_mapped[2] ) ]
        bbox_bottom_mapped[0] = (max_vertical_clip - bbox_bottom_mapped[0]) * ( image_dimensions - 1 ) / max_vertical_clip
        bbox_bottom_mapped[1] = (max_horizontal_clip - bbox_bottom_mapped[1]) * ( image_dimensions - 1 ) / max_vertical_clip

        if bbox_bottom_mapped[0] < image_dimensions and bbox_bottom_mapped[1] < image_dimensions and bbox_bottom_mapped[0] > 0 and bbox_bottom_mapped[1] > 0:
            inv_masked_img[
                int(bbox_bottom_mapped[0] - template_car.shape[0]):int(bbox_bottom_mapped[0]), 
                int(bbox_bottom_mapped[1] - template_car.shape[1]/2):int(bbox_bottom_mapped[1] + template_car.shape[1]/2), :] = template_car
        
        # plt.imshow(inv_masked_img)
        return inv_masked_img