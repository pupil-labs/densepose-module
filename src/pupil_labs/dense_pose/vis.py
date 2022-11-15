import logging
import os
from enum import Enum

import cv2
import numpy as np
import pandas as pd

# Set my own logger
logger = logging.getLogger("pl-densepose-vis")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def vis_pose(frame, result, id_part, xy, bbox=True, scores=True, parts=True):
    """Visualize DensePose data on a frame."""
    for i, box in enumerate(result["pred_boxes_XYXY"]):
        box = np.floor(box.cpu().numpy()).astype(np.int32)
        roi = frame[box[1] : box[3], box[0] : box[2]]
        if bbox:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 180, 0), 2)
        if scores:
            # Put the score on the frame
            cv2.putText(
                frame,
                f"{result['scores'][i]:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        if parts:
            labels_bb = result["pred_densepose"][i].labels.cpu().numpy()
            # Resize to bounding box
            labels_bb = cv2.resize(
                labels_bb,
                (box[2] - box[0], box[3] - box[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            # get the background mask (remain as the frame)
            mask_inv = cv2.bitwise_not(labels_bb.astype(np.uint8))
            bkg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # get the mask of the body part and apply a color map to the parts
            fg = labels_bb * 8
            fg = fg.astype(np.uint8)
            fg = cv2.applyColorMap(fg, cv2.COLORMAP_OCEAN)
            fg = cv2.bitwise_and(fg, fg, mask=labels_bb.astype(np.uint8))

            # plot gazed part in a different color
            if (
                id_part is not None
                and xy[0] < box[2]
                and xy[0] > box[0]
                and xy[1] < box[3]
                and xy[1] > box[1]
            ):
                if len(id_part) == 1 and id_part[0] == 0:
                    continue
                # remove 0 from id_part
                id_part = id_part[1:] if id_part[0] == 0 else id_part
                gazed = labels_bb
                gazed[np.isin(labels_bb, id_part, invert=True)] = 0
                gazed[gazed > 0] = 255
                gazed_mask = gazed.astype(np.uint8)
                g = np.stack(
                    [np.zeros_like(gazed_mask), gazed_mask, gazed_mask],
                    axis=2,
                )
                g = cv2.bitwise_and(g, g, mask=gazed_mask.astype(np.uint8))
                inv_mask = cv2.bitwise_not(gazed_mask.astype(np.uint8))
                fg = cv2.bitwise_and(fg, fg, mask=inv_mask)
                # add the gazed part to the foreground
                fg = cv2.add(fg, g)

            # merge the foreground and background
            blended = cv2.add(bkg, fg)

            # Add transparency
            frame[box[1] : box[3], box[0] : box[2]] = blended
            # cv2.addWeighted(roi, 0.3, blended, 0.7, 0)
    return frame


def report(pandas_df, out_dir):
    """This function takes the final pandas dataframe and returns a report
    with the number of frames with each body part gazed at.
    """
    parts = pandas_df["densepose"]
    parts = parts.str.replace("BACKGROUND", "")
    parts = parts.str.replace(",", "", 1)
    parts = parts.str.split(",")
    parts = parts.apply(lambda x: [i for i in x if i])
    parts = parts.apply(lambda x: [i.strip() for i in x])
    parts = [item for sublist in parts for item in sublist]
    while any(" " in s for s in parts):
        parts = [i.split(" ") for i in parts]
        parts = [item for sublist in parts for item in sublist]
    if any("" in s for s in parts):
        for s in parts:
            if s == "":
                parts.remove(s)

    # Count the number of times each part is gazed at
    parts_count = {i: parts.count(i) for i in parts}
    parts_count = dict(
        sorted(parts_count.items(), key=lambda item: item[1], reverse=True)
    )
    # Make parts count into a Pandas
    parts_count = pd.DataFrame.from_dict(parts_count, orient="index")
    parts_count.columns = ["count"]
    parts_count.index.name = "part"
    parts_count = parts_count.reset_index()

    # Save it as a csv
    parts_count.to_csv(os.path.join(out_dir, "parts_count.csv"), index=False)

    # Load the graphs from the assets folder
    base_body = cv2.imread(
        os.path.join(os.path.dirname(__file__), "assets/body_shape.png")
    )
    col_body = cv2.imread(
        os.path.join(os.path.dirname(__file__), "assets/body_shape_coloured.png")
    )
    part_pixels = dict()
    for part in PartsColour:
        part_pixels[part.name] = np.where(np.all(col_body == part.value, axis=-1))

    step = 255 / parts_count["count"].max()
    for i, row in parts_count.iterrows():
        part = row["part"]
        count = row["count"]
        base_body[part_pixels[part]] = (
            255 - (count * step),
            255 - (count * step),
            255 - (count * step),
        )

    logos = base_body[:200, :, :]
    base_body = base_body[200:, :, :]
    base_body = cv2.applyColorMap(base_body, cv2.COLORMAP_HOT)

    # Add the logos
    gazemap = np.concatenate((logos, base_body), axis=0)

    # Add a colorbar
    margin = np.full((gazemap.shape[0], 100, 3), 255, dtype=np.uint8)
    colorbar = np.zeros((255, 50, 3), dtype=np.uint8)
    for i in range(255):
        colorbar[254 - i, :, :] = (255 - i, 255 - i, 255 - i)
    colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_HOT)

    colorbar = cv2.resize(colorbar, (20, gazemap.shape[0]))

    # add values
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 0, 0)
    lineType = 2

    step = gazemap.shape[0] / parts_count["count"].max()
    for i in range(0, parts_count["count"].max(), 25):
        cv2.putText(
            margin,
            "{}".format(parts_count["count"].max() - i),
            (50, int(np.round(i * step))),
            font,
            fontScale,
            fontColor,
            lineType,
        )
    gazemap = np.concatenate((gazemap, margin), axis=1)
    gazemap = np.concatenate((gazemap, colorbar), axis=1)

    # save the gazemap in rgb
    cv2.cvtColor(gazemap, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(out_dir, "gazemap.png"), gazemap)
    return


class PartsColour(Enum):
    """Body parts for the plot colour"""

    TORSO_BACK = (111, 124, 145)
    TORSO_FRONT = (89, 172, 215)
    RIGHT_HAND = (0, 0, 176)
    LEFT_HAND = (22, 110, 127)
    LEFT_FOOT = (135, 170, 222)
    RIGHT_FOOT = (113, 200, 55)
    UPPER_LEG_RIGHT_BACK = (255, 128, 229)
    UPPER_LEG_LEFT_BACK = (80, 45, 22)
    UPPER_LEG_RIGHT_FRONT = (151, 77, 43)
    UPPER_LEG_LEFT_FRONT = (83, 83, 108)
    LOWER_LEG_RIGHT_BACK = (222, 219, 227)
    LOWER_LEG_LEFT_BACK = (255, 204, 170)
    LOWER_LEG_RIGHT_FRONT = (255, 128, 128)
    LOWER_LEG_LEFT_FRONT = (145, 138, 111)
    UPPER_ARM_LEFT_INSIDE = (186, 172, 18)
    UPPER_ARM_RIGHT_INSIDE = (0, 172, 29)
    UPPER_ARM_LEFT_OUTSIDE = (44, 90, 160)
    UPPER_ARM_RIGHT_OUTSIDE = (83, 108, 83)
    LOWER_ARM_LEFT_INSIDE = (133, 110, 127)
    LOWER_ARM_RIGHT_INSIDE = (0, 172, 176)
    LOWER_ARM_LEFT_OUTSIDE = (55, 200, 55)
    LOWER_ARM_RIGHT_OUTSIDE = (233, 221, 175)
    HEAD_RIGHT = (233, 172, 29)
    HEAD_LEFT = (233, 172, 215)
