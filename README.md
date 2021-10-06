# Fireball
Fireball is a Deep Neural Network (DNN) library for creating, training, evaluating, quantizing, and compressing DNN based models across a range of applications. Here is a summary of main features:
- Easily create any network structure using a limited set of fundamental building blocks chained together in a text string.
- Create models for classification, regression, object detection, and NLP applications.
- Add functionality by creating your own "Blocks" and reuse them in your network structure. 
- Define your own layer types or loss functions and use them in the network structure.
- Apply Low-Rank decomposition on layers of your model to reduce the number of network parameters.
- Apply Pruning to the network parameters.
- Apply K-Means quantization on network parameters to further reduce the size of model. 
- Retrain your model after applying low-rank decomposition, pruning, and/or quantization.
- Compress models using arithmetic entropy coding.
- Export the models to ONEX, Tensorflow, or CoreML even after applying low-rank decomposition, pruning, and/or quantization.

## Fireball Documentation
* [Documentation Home](http://kopartifactory.interdigital.com:9090/FireballDocs/1.5.1/)
* [Installation](http://kopartifactory.interdigital.com:9090/FireballDocs/1.5.1/source/installation.html)
* [Fireball Layers](http://kopartifactory.interdigital.com:9090/FireballDocs/1.5.1/source/layers.html)
* [Fireball API](http://kopartifactory.interdigital.com:9090/FireballDocs/1.5.1/source/model.html)

## Installing Fireball
***Note***: Fireball currently works with python 3.6 and 3.7 and uses Tensorflow 1.14. Support for newer python and tensorflow versions are comming soon.
1. Create a virtual environment:
```
python3 -m venv ve
source ve/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
```
2. Install Fireball:

Use this for GPU machines
```
cd fireball
pip install dist/fireball-1.5.1-0.GPU-py3-none-any.whl
```
or for machines with no GPUs:
```
cd fireball
pip install dist/fireball-1.5.1-0.NoGPU-py3-none-any.whl
```

## Playgrounds
The Playgrounds folder contains a set of tutorials explaining how to use Fireball for some common deep learning models such as object detection and NLP tasks.

[Getting started with Fireball Playgrounds](Playgrounds/README.md)

## Creating Dist Files after modifications
From the root of this repo, issue one of the following commands (for **GPU** and **Non-GPU** installations correspondingly):
```
python3 setup.py bdist_wheel --build-number=0.GPU
python3 setup.py bdist_wheel --build-number=0.NoGPU
```
