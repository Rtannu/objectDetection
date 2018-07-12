# objectDetection
object detection in image using python and openCV

## Dependencies
1. openCV
2. numpy
3. argparse

## Contents
This respository contains following files-
1. images- This folder contains input images.
2. MobileNetSSD_deploy.caffemodel-	Convolutional Neural Network
3. MobileNetSSD_deploy.prototxt.txt- prototxt of MobileNetSSD network
4. object_detection.py-This is main file.

## Usage
- Clone the repository-

```
   git clone https://github.com/Rtannu/objectDetection.git
   cd objectDetection
```
- next step to run main.py file-
 ```
 $ python <path of main file> --prototxt <path of prototxt.txt> --model <path of MobileNetSSd_deploy.caffemodel> --image <path of input image>
  ```
  
  ex-
  
  ```
  $ python object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image images/example_01.jpg 
```
  
  ## Results
  ### Sample Image
  ![Screenshot](https://github.com/Rtannu/DigitRecognition/blob/master/Screenshot%20from%202018-07-08%2016:15:01.png)
