
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import cv2
from PatchMatchCuda import PatchMatch
import matplotlib as plt

def main():
    import os
    _path = r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"

    if os.system("cl.exe"):
        os.environ['PATH'] += ';' + _path
    if os.system("cl.exe"):
        raise RuntimeError("cl.exe still not found, path probably incorrect")
    mod = SourceModule("""
       					__global__ void doublify(float *a)
       					{
       						int idx = threadIdx.x + threadIdx.y*4;
       						 a[idx] *= 2;
       					 }
       			""")

    x = cv2.imread("D:\data\stereo\RAFT\Flowers-imperfect\im0.png")
    y = cv2.imread("D:\data\stereo\RAFT\Flowers-imperfect\im1.png")
    z = cv2.imread("D:\data\stereo\RAFT\Flowers-imperfect\im1.png")
    # z = cv2.imread("D:\data\stereo\RAFT\Flowers-imperfect\disp.tiff")

    x = cv2.resize(x, (1280, 1280))
    y = cv2.resize(y, (1280, 1280))
    z = cv2.resize(z, (1280, 1280))

    x = (x / 255).astype(np.float32)
    y = (y / 255).astype(np.float32)
    z = (z / 255).astype(np.float32)

    pm = PatchMatch(x, x, y, y, 3)
    plt.imshow(pm.visualize())

    pm.propagate(iters=10, rand_search_radius=224)

    plt.imshow(pm.reconstruct_avg(img=y, patch_size=1)[:, :, ::-1])

    plt.imshow(x[:, :, ::-1])

    plt.imshow(pm.visualize())

if __name__ == '__main__':
    main()
