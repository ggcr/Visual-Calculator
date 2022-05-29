# Visual Calculator
This repo is an explanation of the microProject of the subject of Computer Vision. A visual arithmetic calculator capable of performing simple operations with hand gestures.

![](https://github.com/ggcr/Visual-Calculator/blob/main/imgs/Captura%20de%20pantalla%202022-05-20%20a%20las%2016.01.24.png)

As you can see, when the box is in red the sign is input and in green the number is input.

## Table of Contents
- [Objective](#Objective)
- [State of the Art](#State)
    - [Approach with image processing techniques](#processing)
    - [Approach with artificial intelligence techniques](#intelligence)
- [Proposal](#proposal)
    - [Hand Segmentation](#segmentation)
         - [Global Otsu Binarization](#otsu)
         - [Morphological Operations](#morph)
    - [Hand Characterization](#characterization)
         - [The problem of 0 and 1](#zeroone)
    - [Classification](#class)
         - [Insufficient data with CNN](#cnn)
- [Conclusions](#conc)

## <a name="Objective"></a> Objective
The main objective of this project is to build a system that allows you to perform arithmetic calculations using hand gestures.

## <a name="State"></a> State of the Art
Computer Vision has come a long way over the years and it is that neural networks have suddenly changed the methodology used in this field, which is why we have two different approaches to solving the problem:

- **<a name="processing"></a>Approach with image processing techniques**

  In this first approach we will focus on being able to recognize integers given a one-digit sign made by hand.

  - *Hand segmentation*
  
    During this pre-processing phase of the image, it must be possible to locate the hand and differentiate it from obstructions that may be caused by other elements such as camera noise and various backgrounds.

  - *Hand characterization*
  
    Once the hand has been segmented, key points must be found to characterize the different gestures used in the system.

- **<a name="intelligence"></a>Approach with artificial intelligence techniques**

  In this approach we will focus on being able to recognize operators using sign language signs, we will associate characters with good levels of accuracy and differentiable to each of the operators.

  - *<a name="Classification"></a>Classification*

    Finally, the system must be able to map between a specific gesture and a value. We will give meaning to gestures.

As can be seen in the enumeration of the processes to be followed, the modern approach with neural networks is subject to all the past of image processing techniques.

Once these processes have been completed, we will be able to implement the generation of a sequential integer calculations pipeline to complete the visual calculator.

## <a name="proposal"></a>Proposal
The system must work in real time on any webcam or image capture device. The upper right area of the images captured by the cam will be used to avoid being subject to many elements that may hinder the process of hand segmentation.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/input.png" alt="drawing" height="400"/>

### <a name="segmentation"></a>Hand segmentation

Once we have the input delimited to a specific area, we need to segment the hand. More specifically, segmentation is a technique that seeks to find different regions within a single image. In our case we want to consider two regions, the bottom and the hand.

#### <a name="otsu"></a>Global Otsu Binarization

Otsu is a global binarization method, which uses the same threshold value for all pixels in the image. With Otsu we can calculate an optimal threshold value k from finding the intensity value of the histogram that minimizes the intraclass variance.

The Otsu binarization method needs an image with a sufficiently separable histogram to be able to do the threshold partition well. That is, the more differentiated our histogram is, the more optimal the k found by otsu will be.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/histo_con_luz.png" alt="drawing" height="200"/>

As can be seen in the histogram, applying otsu will not be efficient as the image is not separable.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/histo_sin_luz.png" alt="drawing" height="200"/>

In this second case, if we apply a direct light source we see how the histogram is clearly separable, therefore, the algorithm imposes the restriction of **working with a direct light source or with a clearly differentiated background**.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/bin.PNG" alt="drawing" height="200"/>

#### <a name="morph"></a>Morphological operations

If we look at the end result of the Otsu shading we can see how internal shadows are formed in the hand produced by the lighting.
We will have to get rid of these areas because they will not be good for future key-point extraction.

First, we tried to apply a dilate and an open with a 7x7 kernel, but as the future key-point extraction is planned, if we dilate on the horizontal areas it would not negatively affect the extraction and we could reduce the area to remove:

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/kernel7_7.png" alt="drawing" height="200"/>

### <a name="characterization"></a>Hand characterization

Once we have the hand correctly segmented, we will calculate the contours (green color) and the Convex Hull (blue color) with the help of the open-cv library.

From here, starting from the Convex Hull, the characteristics (red dots) have been extracted and we have filtered those that have an angle of less than 90 degrees and are at a sufficient distance away from the perimeter of the Convex Hull.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/Captura%20de%20pantalla%202022-05-20%20a%20las%2012.50.22.png" alt="drawing" height="200"/>

Therefore, for the case of 4, for example, 3 points are detected that satisfy that angle is less than 90 and distance greater than 30:

```
number = count(key_p) + 1
```

But if we look, if we apply a more horizontal structuring element, it will not affect in our obtaining characteristic points of the hand.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/kernel21_7.png" alt="drawing" height="200"/>

We can see how when it comes to feature detection this solves the problem of shadows. Where before a point was detected given the presence of a shadow, now with the morphological operation it does not happen to us.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/comparacioombra(1).png" alt="drawing" height="200"/>

#### <a name="zeroone"></a>The problem of 0 and 1
This way of classifying the number led to problems between 0 and 1.
Since the signs of the numbers 0 and 1 do not contain angles less than 90, it did not detect any feature points and therefore assigned the number 1 to the signs of 1 and 0.

This has been fixed by assigning 0 if the area of the Convex Hull is lower than a certain threshold so that the new way to characterize the numbers will be as follows:

```
if area < 30:
    number = 0
else:
    number = count(key_p) + 1
```

### <a name="class"></a>Classification

As for the arithmetic operators we will use the Sign Language MNIST dataset where we can find almost all the existing characters in the alphabet in sign language, we have decided that the letters that work best will be mapped to the arithmetic operators.

A Convolutional Neural Network (CNN) has been implemented in which its goal will be to produce a model that is able to accept as input a 28x28 image and as output a number that will represent the character or label in question, which then it can be interpreted as a mathematical sign.

If we train the Neural Network with 12 epochs and validate it, it gives us an accuracy of 99.8% on the training set and 96.7% on the test set. Normally, this difference in accuracy between the training set and the test set already indicates a small overfitting on the data.

<figure>
  <img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/confussion_mat.png" alt="drawing" height="400"/>
      <em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Individual CNN Accuraccy for each character.</em>
</figure>


#### <a name="cnn"></a>Insufficient data with CNN

At the same time, CNN's results are bad. It is not able to generalize the knowledge obtained from the dataset and as we can see it depends on the position of the hand on the screen he will say one letter or another.
This is because the data is insufficient. And since our dataset is always the same hand with the same lighting and the same background, the neural network will not be able to work on other scenarios, as it will never have seen them.

<img src="https://github.com/ggcr/Visual-Calculator/blob/main/imgs/Captura%20de%20pantalla%202022-05-20%20a%20las%2015.55.29.png" alt="drawing" height="200"/>

## <a name="conc"></a>Conclusions

The main conclusion we can draw is that it would have been a better option to do Transfer Learning and use a trained network with more varied data such as ImageNet, to ensure that CNN is able to generalize properly. It would have been better to train one of us again with a single dataset.
In addition, one could try to use a binarization technique that does not depend so much on having a light source to differentiate the two areas to be separated.

