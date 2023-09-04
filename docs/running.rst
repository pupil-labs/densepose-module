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
---------

You can also provide additional options while running the command. For example, to specify the input and output paths, use the `--input_path` and `--output_path` options. Additionally, you can use the `--device` option to specify the device to be used (e.g., `cpu` or `cuda` for GPU).
Or the size of the gaze circle used to determine the gazed parts.

Feel free to reach out if you have any questions or need further assistance.

.. option:: -h, --help

   Show this help message and exit.

Main Settings
-------------

.. option:: --input_path INPUT_PATH

   Path to the input video file.

.. option:: --output_path OUTPUT_PATH

   Path where to output files.

Recording Settings
------------------

.. option:: --start START

   Start event name, default is recording.begin.

.. option:: --end END

   End event name, default is recording.end.

DensePose Settings
------------------

.. option:: --model MODEL

   Specify the DensePose model to use. 
   Check out :ref:`under-the-hood``

.. option:: --confidence CONFIDENCE

   Confidence threshold for DensePose model.

.. option:: --device DEVICE

   Device to use for inference. Either cuda or cpu, mps does not work.

Visualization Settings
----------------------

.. option:: -p, --vis

   Enable visualization.

Other Settings
--------------

.. option:: -f, --inference

   Compute inference time.

.. option:: -o, --override

   Override flag, do not perform checks. Use it with Core, PI Player recordings.

.. option:: -cs CIRCLE_SIZE, --circle_size CIRCLE_SIZE

   Size of the gaze circle, used to compute touches.

