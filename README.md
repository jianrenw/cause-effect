Could X cause Y? A causality study
============================

This repo contains the code for Could X cause Y? A causality study (CMU 10708 Project 2022).

Executing this code requires Python 3.6 along with the following packages:

 - pandas (tested with version 10.1)
 - sklearn (tested with version 0.13)
 - numpy (tested with version 1.6.2)
 - scipy (tested with version 0.10.)
 - Pytorch (tested with version 1.0.0)

To run the code,

1. [Download the data](https://drive.google.com/drive/folders/1RM0TuCpfAQbidA0GegSX8S25E9j1FiWD?usp=sharing), and run `python data2img.py` for data visualization.
2. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
3. Train the model by running `python train.py`
4. Make predictions on the validation set by running `python predict.py`
5. Run `python score.py` for evaluation

To train and test the CNN, please refer to `train_CNN.ipynb`.