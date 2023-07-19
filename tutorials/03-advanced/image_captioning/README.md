# Image Captioning
The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

![alt text](png/model.png)



## Usage

#### 1. Download the COCO-2014 dataset

#### 2. Preprocessing

```bash
python create_input_files
```

#### 3. Train the model

```bash
python train.py    
```

#### 4. Test the model 

```bash
python sample.py --image='png/example.png'
```

<br>

