import torchvision.transforms.functional as F
import numpy as np
import cv2
from itertools import repeat
class DrivewayDetection( object ):

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def transformImage( img ):
        w, h = img.shape[2], img.shape[1]
        x_center = w // 2
        # crop into center with 2:1 ratio
        img = img[ :, :, x_center - h:x_center + h ]
        # resize to 640x1280
        img = F.resize( img, ( 640, 640*2 ) ).unsqueeze( 0 )
        # normalize
        img = img.float() / 255.0
        
        return img

    @staticmethod
    def postProcessing( da_seg, ll_seg, yolo_threshold:float, kernel_size:float, out_img_shape=(375, 1242) ) -> tuple:
        """
        yolop outputs da_seg as (batch, 2, h, w) and ll_seg as (batch, 2, h, w) image
        
        for da_seg, we want to:
        - keep the 1th channel 
        - threshold it at 0.5
        - perform erosion + dilation on the image
        - zero pad the image to 375x1242
        - convert the image to a binary mask
        
        for ll_seg, we want to:
        - keep the 1th channel
        - threshold it at 0.5
        - zero pad the image to 375x1242
        - convert the image to a binary mask
        
        we assume an inference time batch size of 1
        
        """
        
        da_seg = ( da_seg[0,1,:,:] > yolo_threshold ).numpy().astype( np.uint8 )
        # cv2 image erosion
        da_seg = cv2.erode( da_seg, np.ones( (5,5), np.uint8 ), iterations=2 )
        da_seg = cv2.dilate( da_seg, np.ones( (5,5), np.uint8 ), iterations=2 )
        # zero pad
        da_seg_pad = np.zeros(out_img_shape)
        da_seg_shrink = cv2.resize(da_seg, ( out_img_shape[0]*2, out_img_shape[0] ) )
        da_seg_pad[:, da_seg_pad.shape[1]//2 - da_seg_shrink.shape[1]//2:da_seg_pad.shape[1]//2 + da_seg_shrink.shape[1]//2] = da_seg_shrink
        da_seg_pad = da_seg_pad.astype(bool)
        
        ll_seg = ( ll_seg[0,1,:,:] > yolo_threshold ).numpy().astype( np.uint8 )
        ll_seg = cv2.dilate(ll_seg, np.ones((3,3), np.uint8), iterations=15)
        ll_seg = cv2.erode(ll_seg, np.ones((3,3), np.uint8), iterations=10)
        ll_seg_pad = np.zeros(out_img_shape)
        ll_seg_shrink = cv2.resize(ll_seg, (out_img_shape[0]*2, out_img_shape[0]))
        ll_seg_pad[:, ll_seg_pad.shape[1]//2 - ll_seg_shrink.shape[1]//2:ll_seg_pad.shape[1]//2 + ll_seg_shrink.shape[1]//2] = ll_seg_shrink
        ll_seg_pad = ll_seg_pad.astype(bool)
        
        return da_seg_pad, ll_seg_pad

    # @staticmethod
    # def performDrivewayDetectionInternal( yolo_model, imgs:list, yolo_threshold:float, kernel_size:float, counter:int, out_img_shape=(375, 1242) ) -> np.ndarray:
    #     print("performDrivewayDetectionInternal:", counter)
    #     img = DrivewayDetection.transformImage(imgs[counter])
    #     # img = img.to( 'mps' )
    #     _, da_seg_out,ll_seg_out = yolo_model( img )
    #     da_seg, ll_seg = DrivewayDetection.postProcessing( da_seg_out.cpu(), ll_seg_out.cpu(), yolo_threshold, kernel_size, out_img_shape )
    #     da_ll_3 = np.zeros( ( da_seg.shape[0], da_seg.shape[1], 3 ) )
    #     da_ll_3[:,:,0] = da_seg
    #     da_ll_3[:,:,1] = ll_seg
    #     return da_ll_3

    @staticmethod
    def performDrivewayDetectionInternal( yolo_model, img:np.ndarray, yolo_threshold:float, kernel_size:float, out_img_shape=(375, 1242) ) -> np.ndarray:
        print("performDrivewayDetectionInternal:")
        img = DrivewayDetection.transformImage(img)
        img = img.to( 'mps' )
        _, da_seg_out,ll_seg_out = yolo_model( img )
        da_seg, ll_seg = DrivewayDetection.postProcessing( da_seg_out.cpu(), ll_seg_out.cpu(), yolo_threshold, kernel_size, out_img_shape )
        da_ll_3 = np.zeros( ( da_seg.shape[0], da_seg.shape[1], 3 ) )
        da_ll_3[:,:,0] = da_seg
        da_ll_3[:,:,1] = ll_seg
        return da_ll_3

    @staticmethod
    def performDrivewayDetection( pool, yolo_model, imgs:list, yolo_threshold:float, kernel_size:float, out_img_shape=(375, 1242) ) -> list:
        print("performDrivewayDetection: ", bool(pool))
        # da_ll_3s = pool.starmap(
        #     DrivewayDetection.performDrivewayDetectionInternal, 
        #     zip(
        #         repeat(yolo_model), 
        #         repeat(imgs), 
        #         repeat(yolo_threshold),
        #         repeat(kernel_size),
        #         range(len(imgs)),
        #         repeat(out_img_shape)
        #     )
        # )

        da_ll_3s = []
        for img in imgs:
            da_ll_3s.append(DrivewayDetection.performDrivewayDetectionInternal(yolo_model, img, yolo_threshold, kernel_size, out_img_shape))
        
        return da_ll_3s