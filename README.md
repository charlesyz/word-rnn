# word-rnn
These four scripts take article databases from ScikitLearn and creates Neural Network MLModels for use in the [article-generator](https://github.com/charlesyz/article-generator) IOS app.
The models can also be run in python using keras.

## Overview
The goal of the neural network is to generate text mimicking the style of news articles on certain topics, such as: Medicine, Sports, Politics or for-sale ads.
The model is a Long Short-Term Memory (LSTM) recurrent neural network. LSTM models "remember" the previous inputs in order to better predict the next output. 

In this case, the neural network has 25 inputs (the 25 previous words), two hidden LSTM layers of 200 nodes, then one output layer with size equal to the
number of unique words in the training data (length of the tokenizer). 

These scripts use Keras and Tensorflow to train the model. It uses a Softmax activation function and the Adam optimization algorithm (A version of stochastic gradient descent).

## How to use the scripts

### prep-data.py
This script takes in the input training data, cleans it, and separates it into sequences of the appropriate length (26 words) for use in train-model.py 

For this script, you need sci-kit learn, which also requires NumPy and SciPy. Run `pip install scikit-learn[alldeps]`

Input text can be any plaintext that you want. There are instructions in the script for what to do if you want to use sci-kit learn, or if you want to use your own text. 
To change which sci-kit learn database the script uses, add or remove elements in the array on line 82: 
```python
categories = ['talk.politics.guns']
```

The list of possible training data is at the top of the script. You can put multiple database names in the categories array, and it will train on all of them.

The script will save the sequences in sequences/out_name.txt

### train-model.py
Train model will take in the sequences from prep-data.py then produce a model in h5 (keras) format and a tokenizer in both plist and pickle format. You can change the output names here: 
```python
# Global file names
in_filename = 'sequences/guns.txt'
model_name = 'models/model-guns.h5'
tokenizer_name = 'tokenizers/tokenizer-guns'
```
This library requires Keras, Tensorflow, NumPy, and CoreMLTools
To install, run: 

```
pip install numpy
pip install tensorflow
pip install keras
pip install coremltools
```

### convert-to-coreml.py
convert-to-coreml.py takes in the h5 model from train-model and converts it to a .mlmodel 

If you plan to use the coreml model with article-generator, *compile the model before putting it into your xcode project*. Make sure you have the newest version of xcode and xcode tools, then run in your terminal: 
```
xcrun coremlcompiler compile MyModel.mlmodel MyModel.mlmodelc
```
Note that the compiled model will appear INSIDE the folder MyModel.mlmodelc


More info on how to use the model with [article-generator](https://github.com/charlesyz/article-generator) can be found on it’s github page.
