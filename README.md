![ID-Fireball-logo](Playgrounds/Fireball.png)

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
- Export the models to ONNX, Tensorflow, or CoreML even after applying low-rank decomposition, pruning, and/or quantization.

## Fireball Documentation
* [Installation](https://interdigitalinc.github.io/Fireball/html//source/installation.html)
* [Documentation Home](https://interdigitalinc.github.io/Fireball/html/)
* [Fireball Layers](https://interdigitalinc.github.io/Fireball/html//source/layers.html)
* [Fireball API](https://interdigitalinc.github.io/Fireball/html//source/model.html)

## Playgrounds
The Playgrounds folder contains a set of tutorials explaining how to use Fireball for some common deep learning models such as object detection and NLP tasks.

[Getting started with Fireball Playgrounds](Playgrounds/README.md)

## Authors

* Shahab Hamidi-Rad, InterDigital AI Lab.
