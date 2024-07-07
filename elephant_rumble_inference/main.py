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

# Windows workarounds
if os.name == 'nt':  # Check if running on Windows
    # touch.hub.get_dir() returns a unix-like path
    print("""########### ATTEMPTING WINDOWS WORKAROUNDS ########""")
    cachedir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "torch",
        "hub"
    )
    torch.hub.set_dir(cachedir)
    print(torch.hub.get_dir())

def parse_args():

    usage = r"""Usage:

        elephant-rumble-inference --save-raven data/*.wav 

        elephant-rumble-inference \
            --save-raven --save-scores \
            --load-labels ~/proj/elephantlistening/data/Rumble \
            --save-dir ~/proj/elephantlistening/tmp/aves/2024-06-01 \
            --visualizations-per-audio-file=5 --visualization-duration=60 \
            ~/proj/elephantlistening/data/Rumble/Training/Sounds/*.wav 

        elephant-rumble-inference --help

        -- nice unlabeled rumbles at
            nn03a_20201020_000100.wav at 10:24:00
            nn02d_202001013_000100.wav 06:25:00

            Interesting soundd at CEB1_20111017_00000.wav at 00:20:00
            Same sound at CEB1_20120715_0000.wav also at 00:20:00
    """
    parser = argparse.ArgumentParser(
        description="Find elephant rumbles in an audio clip", usage=usage
    )
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs",
        help="directory to save outputs",
    )
    parser.add_argument(
        "--visualizations-per-audio-file",
        type=int,
        default=0,
        help="visualiztions are slow so be patient if you pick more than 1",
    )
    parser.add_argument(
        "--visualization-duration",
        type=int,
        default=15,
        help="Minutes of audio for a visualization. 15 is nice for wide monitor, 60 is interesting if you don't mind horizontal scrolling",
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
    # model_name = "elephant_rumble_classifier_500_192_2024-07-03T01:27:40.424353_from_train_folder_valloss=5.55.pth"
    model_name = "best.pth" # windows likes short names
    atw = AvesTorchaudioWrapper().to(DEVICE)
    erc = ElephantRumbleClassifier().to("cpu")
    erc.load_pretrained_weights(model_name)
    print(f"Using weights from {erc.model_name}")
    atw.eval()
    erc.eval()
    return atw, erc


def classify_audio_file(afp, audio_file, limit_audio_hours, save_file_path):
    with torch.inference_mode():
        t0 = time.time()
        scores = afp.classify_wave_file_for_rumbles(
            audio_file, limit_audio_hours=limit_audio_hours
        )
        if save_file_path:
            torch.save(scores, save_file_path)
        t1 = time.time()
        print(f"{t1-t0} to classify {audio_file} [limited to {limit_audio_hours} hours]")
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
        score = 0.9  # TODO get the score from the model
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
    if args.visualizations_per_audio_file > 0:
        visualization_dir = save_dir
    return score_file, raven_file, visualization_dir



def get_windows_torch_hub_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "torch", "hub")


def main():
    args = parse_args()
    print(f"Input files: {args.input_files}")
    atw, erc = initialize_models()
    afp = AudioFileProcessor(atw, erc, device=DEVICE)
    for audio_file in args.input_files:
        audio_file_without_path = os.path.basename(audio_file)

        score_file, raven_file, visualization_dir = choose_save_locations(
            args, audio_file
        )
        print(score_file, raven_file, visualization_dir)
        if raven_file and os.path.exists(raven_file):
            print(f"skipping {raven_file} -- already exists")
            print(f"(delete {raven_file} if you want to re-process it")
            continue

        t0 = time.time()
        scores = classify_audio_file(
            afp, audio_file, args.limit_audio_hours, score_file
        )
        t1 = time.time()

        if args.save_raven:
            save_raven_file(audio_file_without_path, scores, raven_file, afp)

        t2 = time.time()

        if visualization_dir:
            visualization_duration_min  = args.visualization_duration
            visualization_duration_secs = args.visualization_duration * 60

            print("Rendering visualizations...")
            if args.load_labels_from_raven_file_folder:
                rfh = RavenFileHelper(args.load_labels_from_raven_file_folder)
                lbls = rfh.get_all_labels_for_wav_file(audio_file_without_path)
            else:
                lbls = []

            rfh = RavenFileHelper()
            continuous_segments = rfh.find_continuous_segments(scores[:, 1] - scores[:, 0] > 0)
            long_enough_segments = rfh.find_long_enough_segments(continuous_segments, n=3)
            interesting_seconds = [afp.score_index_to_time(bt) for bt,et in long_enough_segments]
            from collections import Counter
            # 5 minute spectrograms are easier to handle than hour long ones.
            interesting_times = Counter([int(sec/visualization_duration_secs)*visualization_duration_secs for sec in interesting_seconds])
            for element, count in interesting_times.most_common():
                print(f"{element}: {count}")
            num_vis =0
            with torch.inference_mode():
                for interesting_time, count in interesting_times.most_common():
                    hour   = (interesting_time) // 60 // 60
                    minute = (interesting_time // 60) % 60
                    dttm   = f"{hour:02}:{minute:02}:00"
                    vis_filename = f"{audio_file_without_path}_{dttm}.png"
                    vis_path = os.path.join(visualization_dir,vis_filename)
                    if os.name == 'nt': # windows doesn't allow a filename to have an iso time in it?
                        vis_path = os.path.join(visualization_dir,f"{audio_file_without_path}_{hour:02}_{minute:02}_00.png")
                    AudioFileVisualizer().visualize_audio_file_fragment(
                        f"{audio_file_without_path}, Starting at {dttm}, Classified by {erc.model_name}",
                        vis_path,
                        audio_file,
                        scores[:, 1],
                        scores[:, 0],
                        afp,
                        start_time=interesting_time,
                        end_time=interesting_time+visualization_duration_secs,
                        width = 4 * visualization_duration_min,
                        height = 4,
                        colormap="clean",
                        labels=lbls,
                    )
                    num_vis += 1
                    if num_vis >= args.visualizations_per_audio_file:
                        print(f"only doing {num_vis} visualization per file")
                        break
