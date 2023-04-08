import argparse


def init_parser():
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

    parser.add_argument("-p", "--vis", action="store_true")
    parser.set_defaults(vis=False)

    parser.add_argument("-f", "--inference", action="store_true")
    parser.set_defaults(inference=False)

    parser.add_argument("-o", "--override", action="store_true")
    parser.set_defaults(override=False)
    return parser
