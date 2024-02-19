Installation
============

Fireball currently works with **python 3.8** or later and uses **Tensorflow 2.7** or later.

Installation for Apple Silicon
------------------------------
For apple M1 machines, you need to use conda environment management system. First create a conda environment, activate it, and install the required conda packages:

.. code-block:: bash

    xcode-select --install
    conda update -n base -c defaults conda
    conda create --name fb22 python=3.9 -y
    conda activate fb22
    pip install --upgrade pip setuptools
    conda install pandoc -y
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
