### Object Detection

#### Major Changes Made:
1. Implemented function to read all inputs from source directory.
2. Generated image pyramids from source image.
3. Implemented function to rotate template image for better object detection. 
4. Implemented template matching technique for searching specific objects.
5. Implemented function to output the locations in a csv file.

#### Advantages of applying rotation and using Template Matching. 
- Improved target-object detection, by atleast 50%.
- Reduced detections of false positives by 70%.

#### Limitations:
- Detecting the location of our target object needs improvement.
- Still detects many unintended objects. (false positives)

### Software and Libraries
This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [OpenCV 3.2](https://docs.opencv.org/3.2.0/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

#### Notes
Currently working on finding a better hack/science/math to compute threshold for improved detection. For this purpose, develop branch of this repo is being used.  




