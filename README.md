# CNN LSTM 
Implementation of CNN LSTM with Resnet backend for image sequence Classification

# Getting Started
## Prerequisites
* PyTorch (ver. 0.4+ required)
* Python 3

### Try on your own dataset 

```
mkdir data
mkdir data/video_data
```
Put your image sequences dataset inside data/image_data
It should be in this form -->
```
+ data 
    + image_data    
            - bowling
            - walking
            + running 
                    - running0
                    - running
                    - runnning1
                        - frame_0.jpg
                        - frame_1.jpg
                        - frame_2.jpg
                          ...
                        - frame_N.jpg
```
## Train
Once you have created the dataset, start training ->
```
python cnn-lstm.py 
```
