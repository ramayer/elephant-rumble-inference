#!/usr/bin/env python
import argparse
import os
import tempfile
import time
import torch
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier
from .audio_file_processor import AudioFileProcessor
from .audio_file_visualizer import AudioFileVisualizer
from .raven_file_helper import RavenFileHelper
from .raven_file_helper import RavenLabel

# consider: https://www.youtube.com/watch?v=Qw9TmrAIS6E for demos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO - try Apple MPS

print(f"using {DEVICE}")
if DEVICE == "cpu":
    print(
        "WARNING - this can be extremely slow on the CPU - prepare to wait a long time"
    )
    print("   Recommend running with --limit-audio-hours=1 when on CPU.")


def parse_args():

    usage = r"""Usage:

        elephant-rumble-inference data/*.wav --save-raven --save-vis 

        elephant-rumble-inference data/*.wav --help # for more options
    """
    parser = argparse.ArgumentParser(
        description="Find elephant rumbles in an audio clip", usage=usage
    )
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(tempfile.gettempdir(), "elephant-rumble-inference"),
        help="directory to safe outputs",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualizations to files",
    )
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Save classification scores to a file",
    )
    parser.add_argument(
        "--save-raven",
        action="store_true",
        help="Save a raven file with found labels",
    )
    parser.add_argument(
        "--load-labels-from-raven-file-folder",
        type=str,
        help="show labels from existing raven files",
    )
    parser.add_argument(
        "--discrete-colormap",
        action="store_true",
        help="Make output visualization use a discrete color map",
    )
    parser.add_argument(
        "--limit-audio-hours",
        type=int,
        default=24,
        help="Limit audio hours (default: 24) (Recommend setting to 1 for CPU).",
    )
    args = parser.parse_args()
    return args


def initialize_models():
    model_name = "best_using_more_varied_training_data"
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-29T22:51:14.720487_valloss=5.83.pth'
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-30T02:01:33.715741_valloss=6.76.pth'
    # model_name = 'elephant_rumble_classifier_500_192_2024-06-30T02:22:33.598037_valloss=6.55.pth'
    model_name = (
        "elephant_rumble_classifier_500_192_2024-06-29T23:39:01.415771_valloss=5.83.pth"
    )
    atw = AvesTorchaudioWrapper().to(DEVICE)
    erc = ElephantRumbleClassifier().to("cpu")
    erc.load_pretrained_weights(model_name)
    atw.eval()
    erc.eval()
    return atw, erc


def classify_audio_file(afp, audio_file, limit_audio_hours, save_file_path):
    scores = afp.classify_wave_file_for_rumbles(
        audio_file, limit_audio_hours=limit_audio_hours
    )
    if save_file_path:
        torch.save(scores, save_file_path)
    return scores


def save_raven_file(audio_file, scores, raven_file, afp):
    rfh = RavenFileHelper()
    continuous_segments = rfh.find_continuous_segments(scores[:, 1] - scores[:, 0] > 0)
    long_enough_segments = rfh.find_long_enough_segments(continuous_segments, n=3)
    print(
        f"of the {len(continuous_segments)} segments classified as rumbles ",
        f"only {len(long_enough_segments)} were over a second long.",
    )
    raven_labels = []
    for s0, s1 in long_enough_segments:
        bt = afp.score_index_to_time(s0)
        et = afp.score_index_to_time(s1)
        lf, hf = 5, 250
        duration = et - bt
        tag1 = tag2 = tag3 = notes = "generated_by_classifier"
        score = "1"  # TODO get the score from the model
        ravenfile = "classifier_generated_raven_file.raven"
        rl = RavenLabel(
            bt,
            et,
            lf,
            hf,
            duration,
            audio_file,
            tag1,
            tag2,
            tag3,
            notes,
            score,
            ravenfile,
        )
        raven_labels.append(rl)
    rfh.write_raven_file(raven_labels, raven_file)


def choose_save_locations(args, audio_file):
    audio_file_without_path = os.path.basename(audio_file)
    save_dir = args.save_dir
    score_file = raven_file = visualization_dir = None
    os.makedirs(save_dir, exist_ok=True)
    if args.save_scores:
        score_file = os.path.join(save_dir, audio_file_without_path + ".scores.pt")
    if args.save_raven:
        raven_file = os.path.join(save_dir, audio_file_without_path + ".raven.txt")
    if args.save_vis:
        visualization_dir = save_dir
    return score_file, raven_file, visualization_dir


def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Input files: {args.input_files}")
    atw, erc = initialize_models()
    afp = AudioFileProcessor(atw, erc, device=DEVICE)
    for audio_file in args.input_files:
        audio_file_without_path = os.path.basename(audio_file)

        score_file, raven_file, visualization_dir = choose_save_locations(
            args, audio_file
        )
        print(score_file, raven_file, visualization_dir)

        t0 = time.time()
        scores = classify_audio_file(
            afp, audio_file, args.limit_audio_hours, score_file
        )
        t1 = time.time()

        if args.save_raven:
            save_raven_file(audio_file_without_path, scores, raven_file, afp)

        t2 = time.time()

        if visualization_dir:
            print("Rendering visualizations...")
            if args.load_labels_from_raven_file_folder:
                rfh = RavenFileHelper(args.load_labels_from_raven_file_folder)
                lbls = rfh.get_all_labels_for_wav_file(audio_file_without_path)
            else:
                lbls = []
            for hour in range(args.limit_audio_hours):
                vis_filename = f"{audio_file_without_path}_{hour:02}:00:00.png"
                vis_path = os.path.join(visualization_dir,vis_filename)
                AudioFileVisualizer().visualize_audio_file_fragment(
                    f"{audio_file_without_path}, Starting at {hour:02}:00:00, Classified by {erc.model_name}",
                    vis_path,
                    audio_file,
                    scores[:, 1],
                    scores[:, 0],
                    afp,
                    start_time=hour * 60 * 60,
                    end_time=(hour + 1) * 60 * 60,
                    make_discrete=args.discrete_colormap,
                    width=199,
                    height=8,
                    labels=lbls,
                )
