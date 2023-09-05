.. _running:

*******************
Running the Module:
*******************

To run the `pupil-labs-dense-pose` module, execute the following command:

.. code-block:: bash

   pl-densepose


Checking the arguments
----------------------

.. code-block:: bash

   pl-densepose -h


Arguments
---------

We can't build a shoe that fits everyone, so we also allow you to pass arguments to the code.

For example, to specify the input and output paths, use the `--input_path` and `--output_path` options. Additionally, you can use the `--device` option to specify the device to be used (e.g., `cpu` or `cuda` for GPU).
Or the size of the gaze circle used to determine the gazed parts.

.. option:: -h, --help

   Show the help message with the arguments.

Path Settings
-------------

If none are given, a UI will open to select the input and output paths (this requires tkinter), so it might not work on your python installation if it is not 
available (e.g. you installed python from homebrew).

The input path shall be the subdirectory of the raw download, containing the video, world, and gaze data. 
The output path shall be the directory where the output files shall be saved.

.. option:: --input_path INPUT_PATH

   Path to the input video file, this should point to the folder containing your recording.

.. option:: --output_path OUTPUT_PATH

   Path where to output files, where do you want the module to save the video and csv files.

Recording Settings
------------------

If you want to run it only on one specific section of your recording, you can pass the start and end event annotations to be used, like this:

.. option:: --start START

   Start event name, default is recording.begin.

.. option:: --end END

   End event name, default is recording.end.

DensePose Settings
------------------

.. option:: --model MODEL ["DensePose_ResNet101_FPN_s1x-e2e.pkl"]

   Specify the DensePose model to use. 
   Check out :doc:`under-the-hood` page for more information.

.. option:: --confidence CONFIDENCE [0.7]

   Confidence threshold for DensePose model. Default is 0.7.

.. option:: --device DEVICE ["cpu"]

   Device to use for inference. Either `cuda` or `cpu`, `mps` does not work.

Visualization Settings
----------------------

.. option:: -p, --vis

   Use the flag --vis to enable live visualization of the output. By default, the visualisation is turned off to save resources, but even with this off, you'll still get the final video output.

Other Settings
--------------

.. option:: -f, --inference

   Compute inference time, this flag will try to estimate the inference time over 100 frames. Default is False.

.. option:: -o, --override

   Override flag, do not perform checks. This flag allows you to use it with Pupil Invisible players that have been loaded onto Pupil Player or even Pupil Core recordings, this flag is experimental, use at your own risk.

.. option:: -cs CIRCLE_SIZE, --circle_size CIRCLE_SIZE [50]

   Size of the gaze circle, used not only for the visualisation but also to compute gazed parts as touches.
   Default value is 50 px.

