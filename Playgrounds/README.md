# Fireball Playgrounds
This folder contains a set of tutorials that can help you get started with Fireball for creating, training, evaluating, and  "playing" with different deep learning models.

## Using the playgrounds
Each playground is a jupyter notebook file organized in different directories. Most of these notebook files ***need GPU*** devices and therefore your notebook server needs to run on a GPU machine with at least 4 Nvidia GPU cards. Here is how you can start the server on a GPU machine and access and execute the notebooks remotely:

### 1) Follow the instruction in the [intallation documentation](https://interdigitalinc.github.io/Fireball/html/source/installation.html).

### 2) The playgrounds use the following datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [ImageNet](http://image-net.org/index)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
- [COCO](https://cocodataset.org)
- [GLUE](https://gluebenchmark.com)
- [RadioML](https://www.deepsig.ai/datasets)


Some of the above datasets require some pre-processing before they can be used by Fireball. Please refer to the corresponding python files in the ```fireball/datasets``` folder for more information on how to prepare these datasets.

### 3) Start notebook server:
```
jupyter notebook --ip <IP_ADDRESS>
```
Replace the <IP_ADDRESS> with the IP address of your GPU machine. 

### 4) Open the Jupyter Notebook starting page:
Copy the address printed by the Jupyter Notebook in the terminal, open a browser window, and paste the address in the address bar. 

### 5) Open the [Contents](./Contents.ipynb) notebook to start navigating through "Fireball Playgrounds".

**Note 1**: Some of the tasks in the playgrounds can take a long time to complete. If you don't want to lose you notebook server if the connection to the server is lost for any reason, you can run the notebook server in a ```tmux``` session which keeps it running when your ssh connection closes.

**Note 2**: In the first cell of each notebook file, there is a "gpus" variable that specifies which GPUs to use. If you want to use a different set of GPU cards, please modify this line accordingly before starting the process.
