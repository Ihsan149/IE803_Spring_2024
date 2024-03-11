Code hosted: https://people.ucsc.edu/~jwang402/car_det/

# Car Detection with Pytorch/Resnet34
---
![cardet](https://people.ucsc.edu/~jwang402/page_resources/img/car_det.png)

---
Requires Jupyter Notebooks support; developed and tested on Google Colab with GPU resources enabled.

## Usage 
#### Run
In Google Colab, ensure GPU runtime is enabled, then run each cell in order.

The program will download all training, validation, and testing data, as well as labels for the data.
The program will then preprocess all data by data in the labels; retraining of the resnet-34 model will then commence.

The model will be loaded for testing, where the the cell will randomly choose a test image to test.

#### General Info
The dataset, https://ai.stanford.edu/~jkrause/cars/car_dataset.html, consists of 16,185 images of cars divided into 196 classes. 8,144 of the images are training images and 8,041 are testing images. Each class is a specific year, make, and model of a car; each of these classes is mapped to a numerical value spanning from 1 to 196. Each numerical class value is mapped to a string in the format [Make Model Year] i.e.: Hyundai Sonata 2012.

The data is labeled by a series of matlab files with the coordinates for a bounding box, its corresponding class number, and filename. The coordinates for the bounding box for each image in pixels are important as some images are car dealer advertisements with extraneous elements in the image such as dealer logos and other advertising elements and watermarks.

![crop](https://people.ucsc.edu/~jwang402/page_resources/img/crop.png)

The retrained resnet 34 model achieves an accuracy of over 80%. The nn takes an image input and passes it through a convolution and pooling layer. The image input is first transformed to a perfect square and are all randomly rotated, flipped and shuffled. The image is then passed into 34 additional layers of similar convolution. Each layer performs a 3x3 convolution with a fixed feature dimension map of values ranging from 64, 128, 256, and 512. The algorithm uses skip connection to bypass the input every 2 convolutions. The general idea behind skip connection is that it adds the output from an earlier layer to a later layer which helps to mitigate the vanishing gradient problem by allowing an alternate shortcut path for the gradient to flow through. All of this also helps the network perform better by allowing the model to learn an identity function which will ensure that the higher layer will perform at least as good as the lower layers and not any worse. We trained the model based on the cropped images provided by our bounding box and utilized GPU hardware (NVIDIA CUDA) for the training.

![crop](https://people.ucsc.edu/~jwang402/page_resources/img/rolls.png)


#### References
Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 2013. 3D Object Representations for Fine-Grained Categorization. 4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia.

Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of Tricks for Image Classification with Convolutional Neural Networks. https://arxiv.org/pdf/1812.01187v2.pdf.

Ross Girshick, Jeff Donahue. Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. https://arxiv.org/pdf/1311.2524.pdf.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual Learning for Image Recognition. Cornell University. 
        https://arxiv.org/abs/1512.03385.
