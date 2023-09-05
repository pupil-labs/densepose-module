.. _installation:

*************
Installation
*************

Below, you'll find instructions on how to install this module/package on your local machine. 
Please note that these instructions may be subject to modifications, and they are provided without any guarantees. 

The latest version have been successfully tested with a Python environment running version 3.11 and PyTorch 2.0.

.. warning:: ⚠️ Windows OS Not Supported

   This package does ``not currently support Windows``, you might be able to run it but it is not tested, and per `detectron2's installation guidelines <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`_ , Windows is not officially supported.


Follow these steps to install and use the `pupil-labs-dense-pose` module:

1. Open a terminal window.

2. Run the following commands (change the python version to the one you have installed):

MacOS (Python 3.11)
-------------------
In MacOS we can only use the CPU version of detectron2, when installing it from Meta's repository there are some issues that you may find. So we will use @johnnynunez's version of detectron2 that works with the latest pytorch version.

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools
      
      # Install torch and torchvision
      pip install torch==2.0.1 torchvision==0.15.2

      # Now, we install detectron2, Meta hasn't update it to run with the latest pytorch version, but thanks to @johnnynunez
      # we have a version that works with the latest. Grab the wheels for your version at https://github.com/johnnynunez/detectron2/actions/runs/5953527699
      # and install them with pip, you will need to point to the wheel you downloaded, e.g.:

      pip install detectron2-0.7-cp311-cp311-macosx_10_9_universal2.whl

      # This will also avoid issues with poetry from python, giving you errors with torch module not being found even though it is installed.
      # Now, we install densepose

      export FORCE_CUDA="0" # as we don't have CUDA
      pip install git+https://github.com/johnnynunez/detectron2@main#subdirectory=projects/DensePose

      # Now we install the module
      python -m pip install 'git+https://github.com/pupil-labs/densepose-module.git'
      # And that's it!

Linux (Python 3.11)
-------------------
On Linux we can either run inference on the CPU or the GPU (if we have CUDA installed). If you want to run it on the CPU, follow these steps:

CPU:
----

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools
      
      # Install torch and torchvision
      pip install torch==2.0.1 torchvision==0.15.2

      # Now, we install detectron2, Meta hasn't update it to run with the latest pytorch version, but thanks to @johnnynunez
      # we have a version that works with the latest. Grab the wheels for your version at https://github.com/johnnynunez/detectron2/actions/runs/5953527699
      # and install them with pip, you will need to point to the wheel you downloaded, e.g.:

      pip install detectron2-3.11-pytorch2.0.1-ubuntu-latest-wheel.whl

      export FORCE_CUDA="0" # as we don't have CUDA
      pip install git+https://github.com/johnnynunez/detectron2@main#subdirectory=projects/DensePose

      # Now we install the module
      python -m pip install 'git+https://github.com/pupil-labs/densepose-module.git'
      # And that's it!

GPU:
----

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools
      
      # Install torch and torchvision
      pip3 install torch+cu torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

      # Now, we install detectron2, Meta hasn't update to run with the latest pytorch version, but thanks to @johnnynunez
      # we have a version that works with the latest. There are wheels for cuda 11.8 and pytorch 2.01 at 
      #(https://app.circleci.com/pipelines/github/facebookresearch/detectron2/2924/workflows/9f85ee27-173e-494c-b699-8ceb110a3398/jobs/14336/artifacts)
      # if you use a different version you will need to build it yourself.

      pip install detectron2-0.7-cp311-cp311-linux_x86_64.whl
      #or to try building your own wheels:
      pip install git+https://github.com/johnnynunez/detectron2.git

      export FORCE_CUDA="1" # as we want to use CUDA
      # We might also need to specify the CUDA home directory
      # like export CUDA_HOME="/usr/local/cuda-11.8"

      pip install git+https://github.com/johnnynunez/detectron2@main#subdirectory=projects/DensePose

      # Now we install the module
      python -m pip install 'git+https://github.com/pupil-labs/densepose-module.git'
      # And that's it!

After is installed properly, you should see a new package  :py:mod:`pupil_labs.dense_pose`.
