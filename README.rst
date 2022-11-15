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


This project allows you to use DensePose from detectron2 together with a Pupil Invisible recording.
It will generate a new visualisation with denseposes overlaid on the video and the gaze on top.
Will also generate a new csv file with the body parts gazed and body heatmap.

To install it:

..  code-block:: python

    pip install pupil-labs-dense-pose

To run it:

..  code-block:: python

    pl-densepose

You can pass the --input_path and --output_path to specify the input and output paths in the command line, alternatively a UI will be prompt to request them.
You can also specify the --device to be used, default is cpu but you can use cuda if you have a GPU.

Check docs/description.rst for a description of arguments, and where is inference happening.
