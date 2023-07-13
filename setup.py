import sys

# Get version from "fireball/__init__.py":
with open("fireball/__init__.py") as f: lines = f.read()
fbVersion = lines.split('\n',1)[0].split("'")[1]

m1Installation = False
if 'bdist_wheel' in sys.argv:
    m1Installation = any(['m1' in x.lower() for x in sys.argv])
    print("\n>>> Making %sInstallation files ...\n"%('M1 ' if m1Installation else ''))
    
else:
    import platform
    import subprocess
    if platform.system().lower() == "darwin":
        processor = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode('utf-8')
        if "apple m1" in processor.lower():
            m1Installation = True

import setuptools

if m1Installation:
    installedPackages = [ "tensorflow-macos==2.8.0",
                          'numpy==1.23.5',
                          "tensorflow-metal==0.4.0",
                          'pyyaml==6.0',
                          'opencv-python==4.5.5.64',
                          'coremltools==5.2.0',
                          'onnx==1.11.0',
                          "onnxruntime-silicon==1.11.1",
                          'matplotlib',
                          'pillow==9.1.0',
                          'notebook',
                          'netron' ]
else:
    installedPackages = [ "tensorflow==2.8.0",
                          'numpy==1.23.5',
                          'pyyaml==6.0',
                          'opencv-python==4.5.5.64',
                          'coremltools==5.2.0',
                          'onnx==1.11.0',
                          "onnxruntime==1.11.1",
                          'matplotlib',
                          'pillow==9.1.0',
                          'notebook',
                          'netron' ]

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
