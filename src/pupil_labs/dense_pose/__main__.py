import argparse
import glob
import logging
import os
import struct
from fractions import Fraction

import av
import cv2
import numpy as np
import pandas as pd
from pupil_labs.dynamic_content_on_rim.uitools.ui_tools import get_path, get_savedir
from pupil_labs.dynamic_content_on_rim.video.read import get_frame, read_video_ts
from rich.logging import RichHandler
from rich.progress import Progress

import pupil_labs.dense_pose.pose as pose
import pupil_labs.dense_pose.vis as pl_dp_vis

logger = logging.getLogger()
logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)

# Check if they are using a 64 bit architecture
verbit = struct.calcsize("P") * 8
if verbit != 64:
    error = "Sorry, this script only works on 64 bit systems!"
    raise Exception(error)


# Main call function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pupil Labs - Dense Pose")
    parser.add_argument("--input_path", default=None, type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--start", default="recording.begin", type=str)
    parser.add_argument("--end", default="recording.end", type=str)
    parser.add_argument(
        "--model", default="DensePose_ResNet101_FPN_s1x-e2e.pkl", type=str
    )
    parser.add_argument("--confidence", default=0.7, type=float)
    parser.add_argument("--device", default="cpu", type=str)

    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--no-vis", dest="vis", action="store_false")
    parser.set_defaults(vis=False)

    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--no-inference", dest="inference", action="store_false")
    parser.set_defaults(inference=False)

    parser.add_argument("-o", "--override", action="store_true")
    parser.set_defaults(override=False)
    args = parser.parse_args()
    logging.info(
        "[white bold on #0d122a]◎ DensePose Module by Pupil Labs[/]",
        extra={"markup": True},
    )
    logging.info(
        "[white bold on #4267B2]∞ Using detectron's libraries by Meta AI[/]",
        extra={"markup": True},
    )
    if args.override:
        logging.info(
            "[white bold on red]⚠️ Running in override mode, this is not officially supported[/]",
            extra={"markup": True},
        )
    logging.info(f"Arguments: {args}")
    if args.input_path is None:
        args.input_path = get_path(
            "Select the video folder in the raw directory", "world_timestamps.csv", None
        )
    logging.info(f"Input path: {args.input_path}")

    # Format to read timestamps
    oftype = {"timestamp [ns]": np.uint64}

    # Read the timestamps
    world_timestamps_df = pd.read_csv(
        os.path.join(args.input_path, "world_timestamps.csv"), dtype=oftype
    )
    if not args.override:
        events_df = pd.read_csv(
            os.path.join(args.input_path, "events.csv"), dtype=oftype
        )
        gaze_df = pd.read_csv(os.path.join(args.input_path, "gaze.csv"), dtype=oftype)
        files = glob.glob(os.path.join(args.input_path, "*.mp4"))
        if len(files) != 1:
            error = "There should be only one video in the raw folder!"
            raise Exception(error)
        video_path = files[0]
    else:
        events_df = pd.read_csv(
            os.path.join(args.input_path, "annotations.csv"), dtype=oftype
        )
        gaze_df = pd.read_csv(
            os.path.join(args.input_path, "gaze_positions.csv"), dtype=oftype
        )
        video_path = os.path.join(args.input_path, "world.mp4")

    # Read the video
    logging.info(
        "[white bold on #0d122a]Reading video...[/]",
        extra={"markup": True},
    )
    _, frames, pts, ts = read_video_ts(video_path)
    logging.info(
        "[white bold on #0d122a]Reading audio...[/]",
        extra={"markup": True},
    )
    with av.open(video_path) as v:
        if not v.streams.audio:
            logging.warning("No audio stream found!")
            audio_stream_available = False
        else:
            audio_stream_available = True

    if audio_stream_available:
        _, audio_frames, audio_pts, audio_ts = read_video_ts(
            video_path, audio=True, auto_thread_type=False
        )
    ts = (
        world_timestamps_df["timestamp [ns]"]
        if not args.override
        else world_timestamps_df["# timestamps [seconds]"]
    )
    if args.override:
        logging.info(
            "[white bold on red]Renaming some columns...[/]",
            extra={"markup": True},
        )
        gaze_df.rename(
            {
                "gaze_timestamp": "timestamp [ns]",
                "norm_pos_x": "gaze x [px]",
                "norm_pos_y": "gaze y [px]",
            },
            axis=1,
            inplace=True,
        )
        gaze_df["timestamp [ns]"].apply(lambda x: x * 1e9)
        ts = ts * 1e9

    video_df = pd.DataFrame(
        {
            "frames": np.arange(frames),
            "pts": [int(pt) for pt in pts],
            "timestamp [ns]": ts,
        }
    )
    if audio_stream_available:
        audio_ts = audio_ts + world_timestamps_df["timestamp [ns]"][0]
        audio_df = pd.DataFrame(
            {
                "frames": np.arange(audio_frames),
                "pts": [int(pt) for pt in audio_pts],
                "timestamp [ns]": audio_ts,
            }
        )
    logging.info("Merging dataframes")
    merged_video = pd.merge_asof(
        video_df,
        gaze_df,
        on="timestamp [ns]",
        direction="nearest",
        suffixes=["video", "gaze"],
    )
    if audio_stream_available:
        merged_audio = pd.merge_asof(
            audio_df,
            video_df,
            on="timestamp [ns]",
            direction="nearest",
            suffixes=["audio", "video"],
        )
    # Chop, chop, chop! (use only the in between events data)
    if args.start != "recording.begin":
        logging.info(f"Looking for start event: {args.start}")
        if not events_df["name"].isin([args.start]).any():
            raise Exception("Start event not found!")
        else:
            start = events_df[events_df["name"] == args.start]["timestamp [ns]"].values[
                0
            ]
            merged_video = merged_video[merged_video["timestamp [ns]"] >= start]
            if audio_stream_available:
                merged_audio = merged_audio[merged_audio["timestamp [ns]"] >= start]
            logging.info(f"Starting at {args.start}")
    if args.end != "recording.end":
        logging.info(f"Looking for end event: {args.end}")
        if not events_df["name"].isin([args.end]).any():
            raise Exception("End event not found!")
        else:
            end = events_df[events_df["name"] == args.end]["timestamp [ns]"].values[0]
            merged_video = merged_video[merged_video["timestamp [ns]"] <= end]
            if audio_stream_available:
                merged_audio = merged_audio[merged_audio["timestamp [ns]"] <= end]
            logging.info(f"Ending at {args.end}")

    # Read first frame
    with av.open(video_path) as vid_container:
        logging.info("Reading first frame")
        vid_frame = next(vid_container.decode(video=0))
        if audio_stream_available:
            aud_frame = next(vid_container.decode(audio=0))

    num_processed_frames = 0

    if args.override:
        logging.info("Geting gaze coordinates on image from normalized coordinates")
        merged_video["gaze x [px]"].apply(lambda x: x * vid_frame.width)
        merged_video["gaze y [px]"].apply(lambda x: x * vid_frame.height)

    # Get the output path
    if args.output_path is None:
        args.output_file = get_savedir(None, type="video")
        args.out_csv = args.output_file.replace(
            os.path.split(args.output_file)[1], "densepose.csv"
        )
        args.output_path = os.path.split(args.output_file)[0]
    else:
        args.output_file = os.path.join(args.output_path, "densepose.mp4")
        args.out_csv = os.path.join(args.output_path, "densepose.csv")
    logging.info(f"Output path: {args.output_file}")

    # Get the model ready
    predictor, visualizer, extractor, cfg = pose.setup_config(
        args.confidence, args.device
    )

    # Here we go!
    with av.open(video_path) as video, av.open(video_path) as audio, av.open(
        args.output_file, "w"
    ) as out_container:
        logging.info("Ready to process video")
        # Prepare the output video
        out_video = out_container.add_stream("libx264", rate=30, options={"crf": "18"})
        out_video.width = video.streams.video[0].width
        out_video.height = video.streams.video[0].height
        out_video.pix_fmt = "yuv420p"
        out_video.codec_context.time_base = Fraction(1, 30)
        if audio_stream_available:
            out_audio = out_container.add_stream("aac", layout="stereo")
            out_audio.rate = audio.streams.audio[0].rate
            out_audio.time_base = out_audio.codec_context.time_base
        lpts = -1
        # For every frame in the video
        with Progress() as progress_bar:
            video_task = progress_bar.add_task(
                "📹 Processing video", total=merged_video.shape[0]
            )
            poses_task = progress_bar.add_task(f"🤸‍♀️ Estimating poses")
            while num_processed_frames < merged_video.shape[0]:
                row = merged_video.iloc[num_processed_frames]
                # Get the frame
                vid_frame, lpts = get_frame(video, int(row["pts"]), lpts, vid_frame)
                if vid_frame is None:
                    break
                img_original = vid_frame.to_ndarray(format="rgb24")
                # Prepare the frame
                frame = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
                frame = np.asarray(frame, dtype=np.float32)
                frame = frame[:, :, :]
                xy = row[["gaze x [px]", "gaze y [px]"]].to_numpy(dtype=np.int32)
                # Get the densepose data
                if args.inference and num_processed_frames == 0:
                    import torch

                    if torch.cuda.is_available():
                        logging.info("Creating logger for inference times")
                        starter, ender = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        repetitions = 400
                        timings = np.zeros((repetitions, 1))
                    else:
                        error = (
                            "CUDA not available, disable inference with --no-inference"
                        )
                        logging.error(error)
                        raise Exception(error)
                elif not args.inference and num_processed_frames == 0:
                    starter, ender, timings = None, None, None

                (
                    frame,
                    _,
                    id_name,
                    starter,
                    ender,
                    timings,
                    poses_task,
                ) = pose.get_densepose(
                    frame,
                    predictor,
                    visualizer,
                    extractor,
                    cfg,
                    xy,
                    starter,
                    ender,
                    timings,
                    frameid=num_processed_frames,
                    progress_bar=progress_bar,
                    poses_task=poses_task,
                )  # frame must be BGR

                # Add id_name to the dataframe
                merged_video.loc[num_processed_frames, "densepose"] = id_name
                # make a circle on the gaze
                if not np.isnan(xy).any():
                    cv2.circle(frame, xy, 50, (0, 0, 255), 10)

                # Finally get thje frame ready.
                out_ = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                if args.vis:
                    cv2.imshow("Merged Video", out_)
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                # Convert to av frame
                cv2.cvtColor(out_, cv2.COLOR_BGR2RGB, out_)
                np.expand_dims(out_, axis=2)
                out_frame = av.VideoFrame.from_ndarray(out_, format="rgb24")
                for packet in out_video.encode(out_frame):
                    out_container.mux(packet)
                if num_processed_frames % 100 == 0:
                    logging.info(
                        f"Processed {num_processed_frames} frames out of {merged_video.shape[0]}"
                    )
                progress_bar.advance(video_task)
                num_processed_frames += 1
                progress_bar.stop_task(video_task)
            for packet in out_video.encode(None):
                out_container.mux(packet)
            if args.inference:
                logging.info(
                    f"Mean inference time: {np.sum(timings) / repetitions } per frame"
                )
                logging.info(f"STD inference time: {np.std(timings)} per frame")
            progress_bar.remove_task(poses_task)
            # audio
            if audio_stream_available:
                audio_task = progress_bar.add_task(
                    "🔊 Processing audio", total=merged_audio.shape[0]
                )
                num_processed_frames = 0
                lpts = -1
                while num_processed_frames < merged_audio.shape[0]:
                    row = merged_audio.iloc[num_processed_frames]
                    aud_frame, lpts = get_frame(
                        audio, int(row["ptsaudio"]), lpts, aud_frame, audio=True
                    )
                    if aud_frame is None:
                        break
                    aud_frame.pts = None
                    af = out_audio.encode(aud_frame)
                    out_container.mux(af)
                    num_processed_frames += 1
                    progress_bar.advance(audio_task)
                progress_bar.stop_task(audio_task)
                for packet in out_audio.encode(None):
                    out_container.mux(packet)
        out_container.close()
        # save the csv
        merged_video.to_csv(args.out_csv, index=False)
        logging.info(f"CSV file saved at: {args.out_csv}")
        # Save the visualisation Gazemap
        pl_dp_vis.report(merged_video, args.output_path)
        logging.info(
            "[white bold on #0d122a]◎ Mischief managed! ⚡️[/]",
            extra={"markup": True},
        )


if __name__ == "__main__":
    main()
