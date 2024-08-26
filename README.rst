.. image:: https://img.shields.io/pypi/v/pupil-labs-dense-pose.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/pupil-labs-dense-pose.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/pupil-labs-dense-pose

.. image:: https://github.com/pupil-labs/densepose-module/workflows/tests/badge.svg
   :target: https://github.com/pupil-labs/densepose-module/actions?query=workflow%3A%22tests%22
   :alt: tests

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: Black

.. .. image:: https://readthedocs.org/projects/skeleton/badge/?version=latest
..    :target: https://skeleton.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/skeleton-2022-informational
   :target: https://blog.jaraco.com/skeleton

Introduction
============

This project allows you to use DensePose from detectron2 together with Pupil Invisible / Neon's recordings.
It generates a new visualization with denseposes overlaid on the video and gaze on top, and also generates a new CSV file with the body parts gazed and body heatmap.

Requirements
============
You should have, Linux or MacOS and Python 3.8 or higher installed on your system.
This file assumes a minimum technical knowledge of the command line and Python, if you are not familiar we recommend that you use our Google Colab notebook instead.

.. image:: https://img.shields.io/static/v1?label=&message=Open%20in%20Google%20Colab&color=blue&labelColor=grey&logo=Google%20Colab&logoColor=#F9AB00
   :target: https://colab.research.google.com/drive/1s6mBNAhcnxhJlqxeaQ2IZMk_Ca381p25?usp=sharing

Installation
============

Follow these steps to install and use the `pupil-labs-dense-pose` module:

On MacOS (using the CPU)

1. Open a terminal window.

2. Run the following commands (change the python version to the one you have installed):

MacOS (Python 3.11)
-------------------
In MacOS we can only use the CPU version of detectron2, when installing it from Meta's repository there are some issues that you may find. So we will use @johnnynunez's version of detectron2 that works with the latest pytorch version.

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools wheel
      
      # Install torch and torchvision
      pip install torch==2.0.1 torchvision==0.15.2

      # Now, we install detectron2, Meta hasn't updated it to run with the latest Pytorch version, but thanks to @johnnynunez
      # we have a version that works with the latest. Grab the wheels for your version at https://github.com/johnnynunez/detectron2/actions/workflows/build-wheels.yml
      # select the latest run for your system and matching Pytorch.
      # Install them with pip, you will need to point to the wheel you downloaded, e.g.:

      pip install detectron2-0.7-cp311-cp311-macosx_10_9_universal2.whl

      # This will also avoid issues with poetry from Python, giving you errors with the torch module not being found even though it is installed.
      # Now, we install densepose

      export FORCE_CUDA="0" # as we don't have CUDA
      pip install git+https://github.com/johnnynunez/detectron2@main#subdirectory=projects/DensePose

      # Now we install the module
      pip install git+https://github.com/pupil-labs/densepose-module
      # And that's it!

Linux (Python 3.11)
-------------------
On Linux, we can run inference on either the CPU or the GPU (if we have CUDA installed). If you want to run it on the CPU, follow these steps:

CPU:
----

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools wheel
      
      # Install torch and torchvision
      pip install torch==2.0.1 torchvision==0.15.2

      # Now, we install detectron2, Meta hasn't update it to run with the latest pytorch version, but thanks to @johnnynunez
      # we have a version that works with the latest. Grab the wheels for your version at https://github.com/johnnynunez/detectron2/actions/workflows/build-wheels.yml
      # and install them with pip, you will need to point to the wheel you downloaded, e.g.:

      pip install detectron2-3.11-pytorch2.0.1-ubuntu-latest-wheel.whl

      export FORCE_CUDA="0" # as we don't have CUDA
      pip install git+https://github.com/johnnynunez/detectron2@main#subdirectory=projects/DensePose

      # Now we install the module
      pip install git+https://github.com/pupil-labs/densepose-module.git
      # And that's it!

GPU:
----

.. code-block:: bash

      # Optional, but recommended, run it on a virtual environment
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -U pip setuptools wheel
      
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
      pip install git+https://github.com/pupil-labs/densepose-module
      # And that's it!


Running the Module
==================

To run the `pupil-labs-dense-pose` module, execute the following command:

.. code-block:: bash

   pl-densepose


Checking the arguments
----------------------

.. code-block:: bash

   pl-densepose -h


Arguments
=========

You can also provide additional options while running the command. For example, to specify the input and output paths, use the `--input_path` and `--output_path` options. Additionally, you can use the `--device` option to specify the device to be used (e.g., `cpu` or `cuda` for GPU).
Or the size of the gaze circle used to determine the gazed parts.

For a detailed description of available arguments and information about where inference is happening, refer to the `docs <http://densepose-module.readthedocs.io/>`_ or our `alpha lab article <https://docs.pupil-labs.com/alpha-lab/dense-pose/>`_

Feel free to reach out if you have any questions or need further assistance.
