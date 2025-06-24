# Camera-Based System for Automatic Pin Mapping

This project provides convenience by distinguishing for IC chips, mapping pin information to images, and fetching circuit drawings and data sheets.

- Source code: https://github.com/Chanjung-K/DLIP_Final_Project/tree/main/Src
- Documentations: https://github.com/Chanjung-K/DLIP_Final_Project/blob/main/Report/DLIP_FinalProject.md



## Environment

- Hardware: Smart phone (used as a webcam via Window11 webcam Connection)

- Python 3.9.23

- OCR model

  easyOCR 1.7.2 **(for CPU user)**

  paddlepaddle-gpu 3.0.0 / paddleOCR 3.0.2 **(Recommended for GPU user)**

- OpenCV - Python 4.10.8
- Levenshteind 0.27.1



## Installation

#### OpenCV

Create virtual environment for Python 3.9.

Name the $ENV as `py39`. If you are in base, enter `conda activate py39`

```
conda create -n py39 python=3.9.12
```

After installation, activate the newly created environment

```
conda activate py39
```

Install Numpy, OpenCV, Matplot, Jupyter

```
conda activate py39
conda install -c anaconda seaborn jupyter
conda install -c anaconda numpy
conda install -c conda-forge opencv
```



#### Levenshtein

You can install the ``Levenshtein`` simply by typing following command in anaconda prompt.

```
pip install Levenshtein
```



#### OCR

##### Installation for GPU user**(recommend)**:

1. **Install paddlepaddle**

Since GPU installation requires specific CUDA versions, the following example is for installing NVIDIA GPU on the Linux platform with CUDA 11.8 & CUDA 12.6. For other platforms, please refer to the instructions in the [PaddlePaddle official installation documentation](https://www.paddlepaddle.org.cn/install/quick).

```
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

2. **Install paddleocr**

```
python -m pip install paddleocr
```

​	For more information, Please refer the [documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html)



##### Installation OCR for CPU user:

1. **Install EasyOCR**

   Open the anaconda prompt, run the environment, and enter it as follows.

```
pip install easyocr
```



## Tutorials

​	Go to following Github link to download our python code, images, and PDFs. You may download either ``easyOCR`` or ``paddleOCR``.

* For those who downloaded ``easyOCR`` (for CPU user), refer to this [link](https://github.com/Chanjung-K/DLIP_Final_Project/blob/main/Src/easyOCR_Module.py).
* For those who downloaded ``paddleOCR`` (for GPU user), refer to this [link.](https://github.com/Chanjung-K/DLIP_Final_Project/blob/main/Src/paddleOCR_Module.py)
* Download our main code. Click [here](https://github.com/Chanjung-K/DLIP_Final_Project/blob/main/Src/main.py).
* Download image and PDFs ``.zip`` file. Click [here](https://github.com/Chanjung-K/DLIP_Final_Project/blob/main/Src/img%20and%20PDF.zip)

Unzip the ``img and PDF.zip`` file. In ``img and PDF.zip`` file, two folders, ``img`` and ``PDF``, should be seen. Redirect them in the same directory of your main and module code. 

Run `Visual Studio Code` open folder where you set all the provided files, and make sure that the files appear well in the `EXPLORER` window.



Open the main code, and edit the importing module section. You must change ``ModuleYouDownloaded`` to either ``paddleOCR_Module`` or ``easyOCR_Module`` as shown below in `main.py`

```python
import cv2 
# Insert appropriate module you downloaded
import paddleOCR_Module as OCRModule
```

Now, you may run the `main.py` code!

If you want to know the operation modes of various command keys and various functions, click on the [Project Overview](#Project Overview) section above for reference.

