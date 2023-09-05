.. _code:

*********
The code:
*********

Please see some brief description of the scripts in this project.



__main__.py
===========
This is the main file that will be run. It will call the other files and run the program.
It re-uses some components of the dynamic-rim module to read the video pts and ts as well as to get the correct video frame for an specific timestamp.
In here you can also see how the audio and video are ts are merged into a single Pandas DataFrame.
The DataFrame is also cropped using the start and end event timestamps.


pose.py
=======
This file contains the main functions to run the densepose.
A setup_config function that will load the config file for the model, as well as the weights.
It also defines the visualizers, the extractor and more importantly the predictor.

These are passed back to main.

.. literalinclude:: ../src/pupil_labs/dense_pose/__main__.py
   :language: python
   :lines: 226-229
   :linenos:

Finally, ``get_densepose`` is the main call that will run the densepose on the video.
Runs the predictor on the frame, which gives the outputs.

The results are a DensePoseChart and PredictionBoxes.

.. literalinclude:: ../src/pupil_labs/dense_pose/pose.py
   :language: python
   :lines: 122-225
   :linenos:

This is called at __main__ here:

.. literalinclude:: ../src/pupil_labs/dense_pose/__main__.py
   :language: python
   :lines: 285-307
   :linenos:

and the predictor, visualizer, config are passed along with the frame, circle_size, and gaze coordinates.

Inference
---------

On `L138 <https://github.com/pupil-labs/densepose-module/blob/eddbceb9ddbb7aa6582c005b588a07e4aa20630c/src/pupil_labs/dense_pose/pose.py#L138>`_ is the call to the predictor and where inference is run.

vis.py
======

vis_pose
--------
A function to visualize the densepose parts, onto video frame.

.. literalinclude:: ../src/pupil_labs/dense_pose/vis.py
   :language: python
   :lines: 10-78
   :linenos:

report
------
Generate a plot and csv file with parts count.

.. literalinclude:: ../src/pupil_labs/dense_pose/vis.py
   :language: python
   :lines: 81-174
   :linenos:
