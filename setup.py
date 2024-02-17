import sys

# Get version from "fireball/__init__.py":
with open("fireball/__init__.py") as f: lines = f.read().split('\n')
fbVersion = '0.0.0'
for line in lines:
    if line[:11]=="__version__":
        fbVersion = line.split("'")[1]
        break

appleSiliconInstallation = False
import platform
if platform.system().lower() == "darwin":
    if 'arm' in platform.processor().lower():
        appleSiliconInstallation = True

import setuptools

if appleSiliconInstallation:
    installedPackages = [ 'matplotlib', 'scikit-learn', 'scipy', 'netron', 'pytest', 'pytest-forked',
                          'numpy', 'pyyaml', 'opencv-python', 'pillow',
                          'notebook==6.5.4', 'traitlets==5.9.0',         # Notebook 6.5.4 (I don't like version 7 yet!)
                          'Sphinx', 'sphinx_rtd_theme', 'nbsphinx',      # For documentation
                          # The following is the best combination that works with all Fireball functionality on Mac as of Feb. 13, 2024
                          "onnx==1.11.0",
                          "tensorflow-macos==2.9.2",
                          "tensorflow-metal==0.5.0",
                          "coremltools==7.1",
                          "onnxruntime==1.16.3",
                        ]
else:
    installedPackages = [ 'matplotlib', 'scikit-learn', 'scipy', 'netron', 'pytest', 'pytest-forked',
                          'numpy', 'pyyaml', 'opencv-python', 'pillow',
                          'notebook==6.5.4', 'traitlets==5.9.0',         # Notebook 6.5.4 (I don't like version 7 yet!)
                          'Sphinx', 'sphinx_rtd_theme', 'nbsphinx',      # For documentation
                          "onnx==1.14.1",
                          "protobuf==3.20.3",
                          "tensorflow==2.13.1",
                          "coremltools==7.1",
                          "onnxruntime==1.16.3"]

setuptools.setup(name="fireball",
                 version = fbVersion,
                 author = "Shahab Hamidi-Rad",
                 author_email = "shahab.hamidi-rad@interdigital.com",
                 description = "Fireball Deep Neural Network Library",
                 long_description = open("README.md", "r").read(),
                 license = open("LICENSE", "r").read(),
                 url = "http://www.interdigital.com",
                 packages = ['fireball','fireball/datasets'],
                 classifiers=[ 'Development Status :: 5 - Production/Stable',
                               'Intended Audience :: Developers',
                               'Topic :: Software Development :: Machine Learning Platform',
                               'Programming Language :: Python :: 3.8',
                               'Programming Language :: Python :: 3.9'],
                 python_requires='>=3.8, <4',
                 install_requires=installedPackages)
