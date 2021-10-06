# Building Fireball Documentation

1. Create a virtual environment:
```
python3 -m venv dve
source dve/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
```

2. Install sphinx and other dependencies:
```
pip install -U Sphinx
pip install sphinx_rtd_theme
pip install tensorflow-gpu==1.14  # or tensorflow==1.14 on Mac
pip install numpy==1.19.0
pip install pyyaml==5.3.1
pip install opencv-python==4.4.0.46
```

3. Then build the html documentation:
```
make clean html
```

The output html is generated in the `_build/html` folder. Open `_build/html/index.html` in your browser to view the locally generated documentation.


