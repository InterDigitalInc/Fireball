Installation
============

Fireball currently works with **python 3.8** or later and uses **Tensorflow 2.7** or later.

Installation for Apple M1
-------------------------
For apple M1 machines, you need to use conda environment management system. First create a conda environment, activate it, and install the required conda packages:

.. code-block:: bash

    conda create --name FB2 python=3.9 -y
    conda activate FB2
    pip install -U pip setuptools wheel
    conda install cmake -y
    conda install -c conda-forge protobuf==3.19.6 -y
    conda install -c apple tensorflow-deps -y
    
Now install fireball:

.. code-block:: bash

    cd Fireball
    pip install -e .

Installation on Unix and MacOS (Intel)
--------------------------------------
First create a virtual environment:

.. code-block:: bash

    python3 -m venv ve
    source ve/bin/activate
    pip install --upgrade pip setuptools

Then install Fireball:

.. code-block:: bash

    cd Fireball
    pip install -e .
