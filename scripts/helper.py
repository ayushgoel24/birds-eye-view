from environment import ApplicationProperties
import numpy as np

class Helper( object ):

    @staticmethod
    def interp2( appProperties: ApplicationProperties, v, xq, yq ):
        
        dim_input = 1
        if len(xq.shape) == 2 or len(yq.shape) == 2:
            dim_input = 2
            q_h = xq.shape[0]
            q_w = xq.shape[1]
            xq = xq.flatten()
            yq = yq.flatten()

        h = v.shape[0]
        w = v.shape[1]
        if xq.shape != yq.shape:
            raise 'query coordinates Xq Yq should have same shape'

        x_floor = np.floor(xq).astype(np.int32)
        y_floor = np.floor(yq).astype(np.int32)
        x_ceil = np.ceil(xq).astype(np.int32)
        y_ceil = np.ceil(yq).astype(np.int32)

        x_floor[x_floor < 0] = 0
        y_floor[y_floor < 0] = 0
        x_ceil[x_ceil < 0] = 0
        y_ceil[y_ceil < 0] = 0

        x_floor[x_floor >= w-1] = w-1
        y_floor[y_floor >= h-1] = h-1
        x_ceil[x_ceil >= w-1] = w-1
        y_ceil[y_ceil >= h-1] = h-1

        v1 = v[y_floor, x_floor]
        v2 = v[y_floor, x_ceil]
        v3 = v[y_ceil, x_floor]
        v4 = v[y_ceil, x_ceil]

        lh = yq - y_floor
        lw = xq - x_floor
        hh = 1 - lh
        hw = 1 - lw

        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw

        interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

        if dim_input == 2:
            return interp_val.reshape(q_h, q_w)
        return interp_val

    @staticmethod
    def make_rot_mat( axis, angle ):
        if axis == 'x':
            return np.array([[1, 0, 0],
                            [0, np.cos( angle ), -np.sin( angle )],
                            [0, np.sin( angle ), np.cos( angle )]])
        if axis == 'y':
            return np.array([[np.cos( angle ), 0, np.sin( angle )],
                            [0, 1, 0],
                            [-np.sin( angle ), 0, np.cos( angle )]])
        if axis == 'z':
            return np.array([[np.cos( angle ), -np.sin( angle ), 0],
                            [np.sin( angle ), np.cos( angle ), 0],
                            [0, 0, 1]])

    @staticmethod
    def parse_calibrations( calib_path ):
        f = open(calib_path, 'r')
        calib = {}
        for line in f:
            line = line.strip()
            if line == '':
                continue
            key, value = line.split(':')
            value = np.array([float(v) for v in value.split()])
            calib[key] = value

        P2 = calib['P2'].reshape(3, 4)
        K = P2[:, :3]
        
        return K

    @staticmethod
    def get_PM( K ):
        R_x90 = Helper.make_rot_mat( 'x', np.pi / 2 )
        R_yn90 = Helper.make_rot_mat( 'y', -np.pi / 2 )
        R_cw = R_yn90 @ R_x90

        t_cw = np.array([0, 1.65, 0.01])

        # transform: rotate camera axes so that z points up, x points forward, y points left
        X_cw = np.hstack((R_cw, t_cw.reshape(3, 1)))
        P_cw = K @ X_cw
        M = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 1]])
        PM = P_cw @ M

        return PM

    