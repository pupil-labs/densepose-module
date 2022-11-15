import pupil_labs.dense_pose.vis as pl_dp_vis
import pandas as pd
import numpy as np
import os

oftype = {"timestamp [ns]": np.uint64}

desktop = os.path.expanduser("~/Desktop")
pd_df = pd.read_csv(os.path.join(desktop, "densepose.csv"), dtype=oftype)
pl_dp_vis.report(pd_df, desktop)
