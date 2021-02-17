# Star Alignment
Align several astronomy / nightscape images by star points in images. Apply to wide angle lens
as well as tele-lens.

A detailed blog describing the motivation and algorithm can be found at [This Site](https://zhuanlan.zhihu.com/p/25311770) 
(which is my special column at zhihu.com, of course, in Chinese)

## matlab code

Check `star_align_average_main.m` as the main script.

## python code

Check `DataModel.py` and run it. Need OpenCV, NumPy, SciPy, pywt, Tyf, matplotlib (optional), tifffile be installed. Matplotlib
is used for debug, and it makes no difference on result if you remove / comment relevant lines.

### **WARNING**
I use [Tyf](https://github.com/Moustikitos/tyf) to handle with metadata of tiff file. But
there are bugs on data types that may cause crashes when reading and writing tiff file.
I have opened an [issue](https://github.com/Moustikitos/tyf/issues/12) and it is not fixed yet.
Relevant lines can be comment out to run the script properly. Or one can correct those 
issues in sources of Tyf packet (I just did that).

Any suggestion about read and write metadata (exif information) of tiff file is welcome.

## Acknowlagements

* PyWavelets: [https://pywavelets.readthedocs.io/en/latest](https://pywavelets.readthedocs.io/en/latest)
* SciPy & NumPy: [https://www.scipy.org/index.html](https://www.scipy.org/index.html)
* OpenCV: [http://opencv.org](http://opencv.org)
* tifffile: [https://github.com/blink1073/tifffile](https://github.com/blink1073/tifffile)
* Tyf: [https://github.com/Moustikitos/tyf](https://github.com/Moustikitos/tyf)

## Some updates by Sean Liu
- Added Python3 support - and all subsequent changes are only for Python3
- Added raw support by rawpy
- Separated command line from DataModel.py
- Removed Tyf import
- The main command line is now alignStar.py
* python3>$ python3 alignStar.py -h
usage: alignStar.py [-h] [-o OUTPUT] [-d] [-k] -f FOCAL [-c CROPFACTOR]
                    images [images ...]

Align stars in the sky

positional arguments:
  images                A list of image files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file
  -d, --debug           Output file
  -k, --keepInterim     Keep all interim images
  -f FOCAL, --focalLength FOCAL
                        Focal length
  -c CROPFACTOR, --cropFactor CROPFACTOR
                        Crop factor (default: 1.0)
- Note that -f argument is now required to supply the focal length without crop factor, as Tyf cannot handle raw files anyway.
- Forego the writing of exif info to final tiff file as it doesn't seem to work well

# 星点对齐叠加
对多张星空图片进行星点对齐并叠加，适用于深空、星野图片，适用于长焦、广角拍摄的图片，改正了常见叠加方法无法对齐广角星空的缺点。

在我的知乎专栏[《星野摄影降噪（2）：对齐叠加》](https://zhuanlan.zhihu.com/p/25311770)中，对算法思路和细节有详细描述，欢迎讨论。

## matlab 代码

脚本 `star_align_average_main.m` 为主脚本。

## python 代码

所有算法都在 `DataModel.py` 文件中，依赖的第三方包：OpenCV, NumPy, SciPy, pywt, Tyf, matplatlib（可选）, tifffile。其中 matplotlib
主要用于调试输出中间图，去掉相关代码对实际功能没有影响。

### **警告**
这里采用了 [Tyf](https://github.com/Moustikitos/tyf) 包来处理 tiff 文件的 exif 信息，
但是这个包有几个关于数据格式的小错误，会导致读取 / 写入信息的时候程序崩溃。
我已经在相关仓库下开了 [issue](https://github.com/Moustikitos/tyf/issues/12)，欢迎讨论。
如果遇到崩溃情况，请把 exif 相关
的行都注释掉，应该就可以正确运行了。或者，也可以手动修改 Tyf 的源文件（我就是这么做的）。

如果有人知道其他能够读写 tiff 文件的 metadata 的包，欢迎进行讨论。

## 感谢

* PyWavelets: [https://pywavelets.readthedocs.io/en/latest](https://pywavelets.readthedocs.io/en/latest)
* SciPy & NumPy: [https://www.scipy.org/index.html](https://www.scipy.org/index.html)
* OpenCV: [http://opencv.org](http://opencv.org)
* tifffile: [https://github.com/blink1073/tifffile](https://github.com/blink1073/tifffile)
* Tyf: [https://github.com/Moustikitos/tyf](https://github.com/Moustikitos/tyf)
