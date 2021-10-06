import sys
gpuInstallation = True
if 'bdist_wheel' in sys.argv:
    gpuInstallation = all(['nogpu' not in x.lower() for x in sys.argv])
    print("\n>>> Making %s Installation...\n"%('GPU' if gpuInstallation else 'NoGPU'))
import setuptools
import fireball

setuptools.setup(name="fireball",
                 version = fireball.__version__,
                 author = "Shahab Hamidi-Rad",
                 author_email = "shahab.hamidi-rad@interdigital.com",
                 description = "My Deep Neural Network Library",
                 long_description = open("README.md", "r").read(),
                 license = open("LICENSE", "r").read(),
                 url = "http://www.interdigital.com",
                 packages = ['fireball','fireball/datasets'],
                 
#                 package_data={'fireball': ['*.pyc']},

                 classifiers=[ 'Development Status :: 5 - Production/Stable',
                               'Intended Audience :: Developers',
                               'Topic :: Software Development :: Machine Learning Platform',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7'],
                               
                 python_requires='>=3.6, <4',
                               
                 install_requires=['tensorflow-gpu==1.14' if gpuInstallation else 'tensorflow==1.14',
                                   'numpy==1.19.0',
                                   'pyyaml==5.3.1',
                                   'opencv-python==4.4.0.46',
                                   'coremltools==3.4',
                                   'onnx==1.7.0',
                                   'onnxruntime==1.5.2',
                                   'matplotlib==3.3.3',
                                   'pillow==8.0.1',
                                   'notebook==6.1.5',
                                   'netron'])
