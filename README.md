# SCGAN-TensorFlow
An implementation of SCGAN using Torch. You can click here to visit [Tensorflow version](https://github.com/tygrer/SCGAN-tensorflow).

Original paper: https://arxiv.org/abs/2210.07594

## Results on test data

### haze -> clear

| Input | Output |
|-------|--------|
|![haze2clear_1](samples/real_apple2orange_1.jpg) | ![haze2clear_1](samples/fake_apple2orange_1.jpg)| 


### clear -> haze

| Input | Output |
|-------|--------|
|![clear2haze_1](samples/real_orange2apple_1.jpg) | ![clear2haze_1](samples/fake_orange2apple_1.jpg)| 


## Prerequisites
- Linux or OSX
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- For MAC users, you need the Linux/GNU commands `gfind` and `gwc`, which can be installed with `brew install findutils coreutils`.

## Getting Started
### Installation
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages `nngraph`, `class`, `display`
```bash
luarocks install nngraph
luarocks install class
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```
- Clone this repo:
```bash
git clone https://github.com/junyanz/CycleGAN
cd CycleGAN
```

## Data preparing
First, download the dataset
* Unpaired dataset: The dataset is built by ourselves, and there are all real haze images from website.

10000 images:
    Address：[Baidu cloud disk](https://pan.baidu.com/s/18Zjm93sZHPHyqgHwlLa-SA)  Extraction code：zvh6
1000  images:
    Address：[Baidu cloud disk](https://pan.baidu.com/s/1BZ2EZS19nYlYEz5J-2Tt6A) Extraction code:47v9 

* Paired dataset: The dataset is added haze by ourselves according to the image depth. 
    
    Address: [Baidu cloud disk](https://pan.baidu.com/s/115OUlSkuYkRUOuGMDREkVg) Extraction code : 63xf

- Now, let's generate dehaze images:
```
DATA_ROOT=./datasets/test name=dehaze_pretrained model=one_direction_test phase=test loadSize=256 fineSize=256 resize_or_crop="scale_width" th test.lua
```
The test results will be saved to `./results/dehaze_pretrained/latest_test/index.html`.  


### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./datasets/download_dataset.sh haze2dehaze
```
- Train a model:
```bash
DATA_ROOT=./datasets/haze2dehaze name=haze2dehaze_model th train.lua
```
- (CPU only) The same training command without using a GPU or CUDNN. Setting the environment variables ```gpu=0 cudnn=0``` forces CPU only
```bash
DATA_ROOT=./datasets/dehaze2haze name=haze2dehaze_model gpu=0 cudnn=0 th train.lua
```
- (Optionally) start the display server to view results as the model trains. (See [Display UI](#display-ui) for more details):
```bash
th -ldisplay.start 8000 0.0.0.0
```

### Test
- Finally, test the model:
```bash
DATA_ROOT=./datasets/haze2dehaze name=haze2dehaze_model phase=test th test.lua
```
The test results will be saved to a html file here: `./results/haze2dehaze_model/latest_test/index.html`.


## Model Zoo
Download the pre-trained models with the following script. The model will be saved to `./checkpoints/model_name/latest_net_G.t7`.


## Display UI
Optionally, for displaying images during training and test, use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

By default, the server listens on localhost. Pass `0.0.0.0` to allow external connections on any interface:
```bash
th -ldisplay.start 8000 0.0.0.0
```
Then open `http://(hostname):(port)/` in your browser to load the remote desktop.


## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2210.07594):

```
@article{zhang2022see,
  title={See Blue Sky: Deep Image Dehaze Using Paired and Unpaired Training Images},
  author={Zhang, Xiaoyan and Tang, Gaoyang and Zhu, Yingying and Tian, Qi},
  journal={arXiv preprint arXiv:2210.07594},
  year={2022}
}
```


## Related Projects:
[CycleGAN](https://github.com/junyanz/CycleGAN)
[pix2pix](https://github.com/phillipi/pix2pix)


## Acknowledgments
Code borrows from [CycleGAN](https://github.com/junyanz/CycleGAN).
