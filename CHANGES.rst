221115 -
Small fixes. Improved report of state in Colab as no progress bar is shown.
Improved gaze visualisation, not all human poses should be highlighted.

221114 - 
Overhaul change in the visualisation and reporting of the data
Removed matplotlib dependency, now everything is done with OpenCV
Results generated are a new video with bbox, and masked parts of the body in blue colors, if the part is gazed, it becomes yellow.
A new csv file is generated with the amount of times gazed on each part of the body.
A new image report shows a heatmap of the gaze on the body.

A  src/pupil_labs/dense_pose/assets/body_shape.png
A  src/pupil_labs/dense_pose/assets/body_shape_coloured.png
A  src/pupil_labs/dense_pose/get_colors.py
M  src/pupil_labs/dense_pose/main.py
M  src/pupil_labs/dense_pose/pose.py
A  src/pupil_labs/dense_pose/vis.py


221019 -
Updated docs, fixed output_path when given via CLI


Initial commit. Barebones working. Input is a raw folder directory containing the video file and the gaze and sections .csv files.
