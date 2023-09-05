.. _under-the-hood:

*****************
Under the hood:
*****************


Contribution:
-------------

We build this module on top of the Detectron2 DensePose project, such that you can now whether and which human body part is gazed on your recordings.
Detectron2 and densepose are from Meta.
This is project is given as it is.

License:
--------

Detectron is released under the `Apache 2.0 license <https://github.com/facebookresearch/detectron2/blob/main/LICENSE>`_.

Acknowledgements:
-----------------

We thank the Meta AI and densepose team, and johnnynunez for implementing pytorch2 support.


Model Weights
-------------

By default, we utilize the following weight configuration: `densepose_rcnn_R_50_FPN_DL_s1x`.

To gain a deeper understanding of this configuration and access other available weights, please refer to the `DensePose README <https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/README.md>`_ on GitHub's Detectron2 project and explore the Model Zoo.

.. seealso::
    See `here <https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose#citing-densepose>`_ how to cite them too.

Why there are no gaze records for the back of the Head, Hands, or Feet?
-----------------------------------------------------------------------

There are no defined parts in DensePose for gaze recording on the back of the head, hands, or feet. Similarly, in the image, the term "frontal view of the arms" pertains to the inside of the arms, rather than the front.
The Detectron2 docs aren't properly updated, but here is a `description of the human part labels. <https://github.com/facebookresearch/detectron2/issues/2185>`_
