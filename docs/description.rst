Main.py
====================
This is the main file that will be run. It will call the other files and run the program.
It re-uses some components of the dynamic-rim module to read the video pts and ts as well as to get the correct video frame for an specific timestamp.
In here you can also see how the audio and video are ts are merged into a single Pandas DataFrame.
The DataFrameis also cropped using the start and end event timestamps.

**input_path**:
The input is a subfolders from the RAW enrichment download that you get from the Cloud. This folder should contain the gaze.csv file, the scene camera video and the events file.

**output_path**:
This is the folder path where the output will be saved.
The output will be a video file with the densepose and a csv file with the merged DataFrame.

**start**:
A string with the name of the starting event. Defaults to recording.begin

**end**:
A string with the name of the ending event. Defaults to recording.end

**model**:
Currently does nothing, the idea is to be able to select the model later passing it in main.py [L174](https://github.com/pupil-labs/densepose-module/blob/eddbceb9ddbb7aa6582c005b588a07e4aa20630c/src/pupil_labs/dense_pose/main.py#L174)
As of now it would take the config file that is in the config dir that is not the Base model.
Check setup_config function in pose.py  for more info.
Would be worth to check the MODEL_ZOO for the best model.

**confidence**:
The confidence threshold for the densepose. Defaults to 0.7

**device**:
The device to run the densepose. Defaults to cuda but can also run on cpu.
No support for MPS yet.

**vis**:
Boolean to show the video with the densepose while running it, good for debug. Defaults to False.
Pass it as --vis or --no-vis

**inference**:
Boolean to run the inference estimation or not. Defaults to True.
Pass it as --inference or --no-inference
[L217](https://github.com/pupil-labs/densepose-module/blob/eddbceb9ddbb7aa6582c005b588a07e4aa20630c/src/pupil_labs/dense_pose/main.py#L217) in main.py defines the number of repetitions for the inference.


pose.py
====================
This file contains the main functions to run the densepose.
A setup_config function that will load the config file for the model, as well as the weights.
It also defines the visualizers, the extractor and more importantly the predictor.

These are passed back to main.

Finally, get_densepose is the main call that will run the densepose on the video.
Runs the predictor on the frame, which gives the outputs.

The results are a DensePoseChart and PredictionBoxes.
The Detectron2 docs aren't properly updated, but here is a description
https://github.com/facebookresearch/detectron2/issues/2185

On [L138](https://github.com/pupil-labs/densepose-module/blob/eddbceb9ddbb7aa6582c005b588a07e4aa20630c/src/pupil_labs/dense_pose/pose.py#L138) is the call to the predictor and where inference is run.
