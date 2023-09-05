.. _under-the-hood:

*****************
Under the hood:
*****************


Contribution:
-------------

License:
--------

Acknowledgements:
-----------------



Model Weights
-------------

By default, we utilize the following weight configuration: `densepose_rcnn_R_50_FPN_DL_s1x`.

To gain a deeper understanding of this configuration and access other available weights, please refer to the [DensePose README](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/README.md) on GitHub's Detectron2 project and explore the Model Zoo.

.. seealso::
    Model

Why Are There No Gaze Records for the Back of the Head, Hands, or Feet?
-----------------------------------------------------------------------

There are no defined parts in DensePose for gaze recording on the back of the head, hands, or feet. Similarly, in the image, the term "frontal view of the arms" pertains to the inside of the arms, rather than the front.
The Detectron2 docs aren't properly updated, but here is a `description of the human parts <https://github.com/facebookresearch/detectron2/issues/2185>`_
