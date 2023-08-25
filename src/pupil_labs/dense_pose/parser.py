import argparse


def group_decorator(title):
    def inner_group_decorator(func):
        def wrapper(parser):
            group = parser.add_argument_group(title)
            func(group)
            return group

        return wrapper

    return inner_group_decorator


def init_parser():
    parser = argparse.ArgumentParser(description="Pupil Labs - Dense Pose")

    @group_decorator("Main settings")
    def main_settings_group(group):
        group.add_argument(
            "--input_path", default=None, type=str, help="Path to the input video file"
        )
        group.add_argument(
            "--output_path", default=None, type=str, help="Path where to output files"
        )

    @group_decorator("Recording settings")
    def rec_settings_group(group):
        group.add_argument(
            "--start",
            default="recording.begin",
            type=str,
            help="Start event name, default is recording.begin",
        )
        group.add_argument(
            "--end",
            default="recording.end",
            type=str,
            help="End event name, default is recording.end",
        )

    @group_decorator("DensePose settings")
    def densepose_settings_group(group):
        group.add_argument(
            "--model", default="DensePose_ResNet101_FPN_s1x-e2e.pkl", type=str
        )
        group.add_argument(
            "--confidence", default=0.7, type=float, help="Confidence threshold"
        )
        group.add_argument(
            "--device", default="cpu", type=str, help="Device to use for inference"
        )

    @group_decorator("Visualization settings")
    def vis_settings_group(group):
        group.add_argument("-p", "--vis", action="store_true")
        parser.set_defaults(vis=False)

    @group_decorator("Other settings")
    def other_settings_group(group):
        group.add_argument(
            "-f", "--inference", action="store_true", help="Compute inference time"
        )
        parser.set_defaults(inference=False)

        group.add_argument(
            "-o",
            "--override",
            action="store_true",
            help="Override flag, do not perform checks, use it with Core, PI Player recordings",
        )
        parser.set_defaults(override=False)

        group.add_argument(
            "-cs",
            "--circle_size",
            default=50,
            type=int,
            help="Size of the gaze circle, used to compute touches",
        )

    main_settings_group(parser)
    rec_settings_group(parser)
    densepose_settings_group(parser)
    vis_settings_group(parser)
    other_settings_group(parser)

    return parser


if __name__ == "__main__":
    parser = init_parser()
