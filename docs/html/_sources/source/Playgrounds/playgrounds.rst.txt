Playgrounds
===========
The Playgrounds folder contains a set of tutorials explaining how to use Fireball for some common deep learning models such as object detection and NLP tasks.

Using the playgrounds
~~~~~~~~~~~~~~~~~~~~~
Playgrounds are a set of **jupyter notebook** files organized in different directories. Most of these notebook files **need GPU** devices and therefore your notebook server needs to run on a GPU machine with at least 4 Nvidia GPU cards. Here is how you can start the server on a GPU machine and access and execute the notebooks remotely:

Initializing Fireball environment
---------------------------------
On your GPU machine create and initialize a virtual environment. If you already have created a virtual environment for fireball, activate it and skip to the next step. Otherwise use the procedure explained in the "Installation" to create and initialize a virtual environment.

Setting up datasets
-------------------
Fireball playgrounds use the following datasets:

* `MNIST <http://yann.lecun.com/exdb/mnist/>`_
* `ImageNet <http://image-net.org/index>`_
* `COCO <https://cocodataset.org>`_
* `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`_

Some of the above datasets require some pre-processing before they can be used by Fireball. Please refer to the corresponding python files in the ``fireball/datasets`` folder for more information on how to prepare these datasets.

Start the notebook server
-------------------------

.. code-block:: bash

    jupyter notebook --ip <IP_ADDRESS>

Replace the *<IP_ADDRESS>* with the IP address of your GPU machine. Then copy the address printed by the Jupyter Notebook in the terminal, open a browser window, and paste the address in the address bar. Now open ``Contents.ipynb`` to start navigating through Fireball Playgrounds.

Notes
-----
* Some of the tasks in the playgrounds can take a long time to complete. If you don't want to lose you notebook server if the connection to the server is lost for any reason, you can run the notebook server in a ``tmux`` session which keeps it running when your ssh connection closes.

* In the first cell of each notebook file, there is a "gpus" variable that specifies which GPUs to use. If you want to use a different set of GPU cards, please modify this line accordingly before starting the process.

* If a required dataset is not available on the machine, the first cell of the playground files tries to download and install the dataset. For some large datasets, this may take a long time. Please be patient and avoid interrupting the process during the download.

