# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd). 
- Note: While I would love it if this were my full-time job, this is currently only a hobby of mine so I cannot guarantee that I will be able to dedicate all my time to updating this repo.  That being said, thanks to everyone for your help and feedback it is really appreciated, and I will try to address everything as soon as I can. 


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#download-voc2007-trainval--test) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training! 
  * To use Visdom in the browser: 
  ```Shell
  # First install Python server and client 
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently only support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/), but are adding [COCO](http://mscoco.org/) and hopefully [ImageNet](http://www.image-net.org/) soon.
- UPDATE: We have switched from PIL Image support to cv2. The plan is to create a branch that uses PIL as well.  

## Datasets
To make things easy, we provide a simple VOC dataset loader that inherits `torch.utils.data.Dataset` making it fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Currently we only support training on v2 (the newest version).
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)
  
## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Performance

#### VOC2007 Test

##### mAP

Because My GPU is GTX1060, it only has 6G memeory.
So I change the batch_size to 24.

--batch_size 24

The bottom was tested in 120000 iterations net paragrams.

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.57 % |

##### Evaluation report for the current version

VOC07 metric? Yes

AP for aeroplane = 0.7996<br />
AP for bicycle = 0.8466<br />
AP for bird = 0.7543<br />
AP for boat = 0.7005<br />
AP for bottle = 0.5104<br />
AP for bus = 0.8485<br />
AP for car = 0.8642<br />
AP for cat = 0.8769<br />
AP for chair = 0.6245<br />
AP for cow = 0.8208<br />
AP for diningtable = 0.7820<br />
AP for dog = 0.8539<br />
AP for horse = 0.8732<br />
AP for motorbike = 0.8473<br />
AP for person = 0.7895<br />
AP for pottedplant = 0.5291<br />
AP for sheep = 0.7550<br />
AP for sofa = 0.7960<br />
AP for train = 0.8688<br />
AP for tvmonitor = 0.7735<br />
Mean AP = 0.7757<br />



The bottom was tested in 110000 iterations net paragrams.

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.67 % |

##### Evaluation report for the current version

VOC07 metric? Yes

AP for aeroplane = 0.7997<br />
AP for bicycle = 0.8461<br />
AP for bird = 0.7607<br />
AP for boat = 0.6971<br />
AP for bottle = 0.5117<br />
AP for bus = 0.8484<br />
AP for car = 0.8655<br />
AP for cat = 0.8751<br />
AP for chair = 0.6240<br />
AP for cow = 0.8331<br />
AP for diningtable = 0.7834<br />
AP for dog = 0.8541<br />
AP for horse = 0.8725<br />
AP for motorbike = 0.8480<br />
AP for person = 0.7892<br />
AP for pottedplant = 0.5308<br />
AP for sheep = 0.7595<br />
AP for sofa = 0.7937<br />
AP for train = 0.8692<br />
AP for tvmonitor = 0.7719<br />
Mean AP = 0.7767<br />



The bottom was tested in 105000 iterations net paragrams.

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.69 % |

##### Evaluation report for the current version

VOC07 metric? Yes

AP for aeroplane = 0.7956<br />
AP for bicycle = 0.8471<br />
AP for bird = 0.7632<br />
AP for boat = 0.7035<br />
AP for bottle = 0.5118<br />
AP for bus = 0.8479<br />
AP for car = 0.8665<br />
AP for cat = 0.8802<br />
AP for chair = 0.6217<br />
AP for cow = 0.8159<br />
AP for diningtable = 0.7941<br />
AP for dog = 0.8569<br />
AP for horse = 0.8728<br />
AP for motorbike = 0.8435<br />
AP for person = 0.7883<br />
AP for pottedplant = 0.5260<br />
AP for sheep = 0.7583<br />
AP for sofa = 0.7967<br />
AP for train = 0.8771<br />
AP for tvmonitor = 0.7700<br />
Mean AP = 0.7769<br />



##### FPS
**GTX 1060:** ~45.45 FPS 

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models: 
    * SSD300 v2 trained on VOC0712 (newest PyTorch version)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 v2 trained on VOC0712 (original Caffe version)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
    * SSD300 v1 (original/old pool6 version) trained on VOC07
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_voc07.tar.gz
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325) 
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run): 
    `jupyter notebook` 

    2. If using [pip](https://pypi.python.org/pypi/pip):
    
```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## TODO
We have accumulated the following to-do list, which you can expect to be done in the very near future
- Still to come:
  * Train SSD300 with batch norm
  * Add support for SSD512 training and testing
  * Add support for COCO dataset
  * Create a functional model definition for Sergey Zagoruyko's [functional-zoo](https://github.com/szagoruyko/functional-zoo)


## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo): 
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow) 
