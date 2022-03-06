# 个人所有关于三维重建的笔记、资料收集、代码

## 0. 立体匹配精度评价

根据自己模组的模型参数，算法精度，得到该参数下的模组的精度。

python脚本：

```python
# Descirption:输入双目模组的相关参数 得到该参数下的模组的精度
# Author:xiaoxiong
# Dara:2022-03-06
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# 1280 800 104 76 60mm 1pixel maxdisparity
class stereoCameraModel(object):
    def __init__(self, iW, iH, hFOV, vFOV, baseline, maxdisparity, matchpercious):
        self.iW = iW  # 图像宽度
        self.iH = iH  # 图像高度
        self.horizontalFOV = hFOV*math.pi/180  # 相机水平方向FOV
        self.verticalFOV = vFOV*math.pi/180  # 相机竖直方向FOV
        self.focus = self.iW*0.5/math.tan(self.horizontalFOV/2)  # 相机焦距
        self.baseline = baseline  # 基线长度
        self.maxDisparity = maxdisparity # 算法最大视差
        self.matchPercious = matchpercious  # 左右目匹配精度
        self.blindMinDis = self.baseline * self.focus / self.maxDisparity  # 可测最小距离

    def print_cameraModelParameter(self):
        print('image width:{}; image height:{}'.format(self.iW, self.iH))
        print('camera horizontal FOV:{} degree; camera vertical FOV:{} degree'.format(self.horizontalFOV*180/math.pi, self.verticalFOV*180/math.pi))
        print('camera focus:{}; Module baseline:{} mm; Module Left Right match Percious:{} pixel'.format(self.focus, self.baseline, self.matchPercious))
        print('camera maxDisparity:{}; Module blindMinDis:{} mm'.format(self.maxDisparity, self.blindMinDis))


# 在实际匹配精度下模组在不同距离的精度
# z = f*b/d
    def getErrorInDistance(self, distance):
        d_current = self.focus*self.baseline/distance  # 当视差为d_current有distance
        error_Z = self.focus*self.baseline/math.fabs(d_current - self.matchPercious)
        delta_Z = math.fabs(error_Z-distance)
        Percious = (delta_Z/distance)*100
        return delta_Z, Percious


# 640宽 大FOV137 基线35mm 0.5个像素匹配精度
camera1 = stereoCameraModel(640, 480, 91, 77, 60, 192, 0.3)


print('camera1 #:')
camera1.print_cameraModelParameter()
for d in np.arange(100, 2200, 200):
    print('distance: {} mm; deltaZ: {} mm; Percious: {} %'.format(d, camera1.getErrorInDistance(d)[0], camera1.getErrorInDistance(d)[1]))
```

示例输出：

>camera1 #:
>image width:640; image height:480
>camera horizontal FOV:91.0 degree; camera vertical FOV:77.00000000000001 degree
>camera focus:314.4631241970208; Module baseline:60 mm; Module Left Right match Percious:0.3 pixel
>camera maxDisparity:192; Module blindMinDis:98.269726311569 mm
>distance: 100 mm; deltaZ: 0.15925437144211685 mm; Percious: 0.15925437144211685 %
>distance: 300 mm; deltaZ: 1.4378690817155189 mm; Percious: 0.4792896939051729 %
>distance: 500 mm; deltaZ: 4.006883836727297 mm; Percious: 0.8013767673454596 %
>distance: 700 mm; deltaZ: 7.8787477014404885 mm; Percious: 1.1255353859200699 %
>distance: 900 mm; deltaZ: 13.066070393024347 mm; Percious: 1.4517855992249276 %
>distance: 1100 mm; deltaZ: 19.581624880715708 mm; Percious: 1.7801477164287007 %
>distance: 1300 mm; deltaZ: 27.438350036331258 mm; Percious: 2.11064231048702 %
>distance: 1500 mm; deltaZ: 36.649353336589456 mm; Percious: 2.4432902224392974 %
>distance: 1700 mm; deltaZ: 47.227913618424054 mm; Percious: 2.77811256578965 %
>distance: 1900 mm; deltaZ: 59.18748388850736 mm; Percious: 3.1151307309740717 %
>distance: 2100 mm; deltaZ: 72.54169418823267 mm; Percious: 3.4543663899158417 %
>
>Process finished with exit code 0

## 1. 半全局立体匹配

### A.SGM



## 2. 全局匹配

### 1. PatchMatch



## 3.深度学习

[详解深度学习三维重建网络：MVSNet、PatchMatchNet、JDACS-MS](https://mp.weixin.qq.com/s/KSv4pk1sGVx1-vWQjW6HFg)





# 二、双目结构光

## 1. 资料

## 2. 笔记



# 三、线激光

