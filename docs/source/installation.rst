Installation
============

Fireball currently works with **python 3.6 and 3.7** and uses **Tensorflow 1.14**. Support for newer python and tensorflow versions are comming soon.

Create a virtual environment
----------------------------

.. code-block:: bash

    python3 -m venv ve
    source ve/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools

Install Fireball
----------------
For GPU machines:

.. code-block:: bash

    cd fireball
    pip install dist/fireball-1.5.1-0.GPU-py3-none-any.whl

or for machines with no GPUs:

.. code-block:: bash

    cd fireball
    pip install dist/fireball-1.5.1-0.NoGPU-py3-none-any.whl
