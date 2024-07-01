#!/usr/bin/env python

import torch
import argparse
import einops
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier
from .audio_file_processor import AudioFileProcessor
from .audio_file_visualizer import AudioFileVisualizer

# consider: https://www.youtube.com/watch?v=Qw9TmrAIS6E for demos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE}")


def parse_args():
    
    usage = """Usage:
       elephant-rumble-inference  /tmp/test.wav \
          --save-classification-scores=/tmp/scores.pt \
          --save-visualizations=/tmp/vis.png
    """

    parser = argparse.ArgumentParser(description="Find elephant rumbles in an audio clip",
                                     usage=usage)
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument("--save-visualizations", type=str, help="Save visualizations to a file (optional)")
    parser.add_argument("--save-classification-scores", type=str, help="Save classification scores to a file (optional)")
    parser.add_argument("--save-raven-file", type=str, help="Specify the name of the Raven file (optional)")
    args = parser.parse_args()
    return args

    parser.add_argument("--save-classification-scores", type=str, help="save")

def initialize_models():
    atw = AvesTorchaudioWrapper().to(DEVICE)
    erc = ElephantRumbleClassifier().to("cpu")
    erc_weights = erc.choose_model_weights("best_using_training_data_only")
    erc.load_pretrained_weights(erc_weights)
    print("Hello world!")
    return atw, erc

def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Input files: {args.input_files}")
    atw, erc = initialize_models()
    afp = AudioFileProcessor(atw, erc,device=DEVICE)
    for audio_file in args.input_files:
        scores = afp.classify_wave_file_for_rumbles(audio_file)
        if args.save_classification_scores:
          torch.save(scores, args.save_classification_scores)
        if args.save_visualizations:
          AudioFileVisualizer().visualize_audio_file_fragment(
              f"{audio_file} and scores",
              args.save_visualizations,
              audio_file,
              scores[:,1],
              scores[:,0],
              afp,
              start_time=0,
              end_time=60*5
          )

