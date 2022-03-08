from pathlib import Path
import numpy as np
import csv
import re
import cv2
from PIL import Image
from libtiff import TIFF


def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)
    return calib

def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #
    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    #
    # show(img, "disparity")

    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib=None):
    dispariy, [shape, scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        # scale factor is used here
        depth_map = (fx * base_line / (dispariy + doffs))
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map)
        depth_map = depth_map.squeeze()

        dispariy = np.reshape(dispariy, newshape=shape)
        dispariy = np.flipud(dispariy)
        dispariy = dispariy.squeeze()

        return dispariy, depth_map


def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)

def getDataFormPfm(pfm_name,calib_name,save_floder):
    # calibration information
    calib = read_calib(calib_name)
    # create depth map
    disp_map_left, depth_map_left = create_depth_map(pfm_name, calib)
    # 生成点云
    fx = float(calib['cam0'].split(' ')[0].lstrip('['))
    cx = float(calib['cam0'].split(' ')[2].rstrip(';'))
    fy = float(calib['cam0'].split(' ')[4])
    cy = float(calib['cam0'].split(' ')[5].rstrip(';'))

    u = np.arange(0, 2888, 1)
    v = np.arange(0, 1984, 1)
    u, v = np.meshgrid(u, v)

    z = depth_map_left #深度图
    x = (u - cx) * z/fx
    y = (v - cy) * z/fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    savenamepointcloud = save_floder.joinpath('pointcloud.txt')
    savenamedepth = save_floder.joinpath('depth.tiff')
    savenamedisp = save_floder.joinpath('disp.tiff')


    imDepth = Image.fromarray(z)
    imDisp = Image.fromarray(disp_map_left)

    np.savetxt(savenamepointcloud, points, fmt='%0.8f')
    print('save {}'.format(savenamepointcloud))

    tif = TIFF.open(savenamedepth, mode='w')
    tif.write_image(imDepth, compression=None)
    tif.close()
    print('save {}'.format(savenamedepth))

    tif = TIFF.open(savenamedisp, mode='w')
    tif.write_image(imDisp, compression=None)
    tif.close()
    print('save {}'.format(savenamedisp))

    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    # im_color=cv2.applyColorMap(cv2.convertScaleAbs(depth_map_left,alpha=15),cv2.COLORMAP_JET)
    # win_name = 'depth'
    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(win_name, im_color)
    # cv2.waitKey()
    # cv2.destroyWindow(win_name)

def main():
    pfm_file_dir = Path(r'D:\data\stereo\RAFT\Flowers-imperfect')
    calib_file_path = pfm_file_dir.joinpath('calib.txt')
    disp_left = pfm_file_dir.joinpath('disp0.pfm')

    getDataFormPfm(disp_left, calib_file_path, pfm_file_dir)

if __name__ == '__main__':
    main()
