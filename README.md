# CNN LSTM 
Implementation of CNN LSTM with Resnet backend for image sequence Classification

# Getting Started
## Prerequisites
* PyTorch (ver. 0.4+ required)
* Python 3

### Try on your own dataset 

```
mkdir data
mkdir data/image_data
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

Set parameters in code cnn-lstm.py -->
```
def main():
    # Parameters
    dataset_dir = 'data/image_data/'
    num_classes = 5
    batch_size = 8
    sequence_length = 24
    img_height = 224
    img_width = 224
    model_dir = 'models'  # Directory to save model checkpoints
    val_split = 0.2  # Fraction of data to use for validation
    num_epochs = 100
    patience = 30
    min_delta = 0

```
## Train
Once you have created the dataset, start training ->
```
python cnn-lstm.py 
```
