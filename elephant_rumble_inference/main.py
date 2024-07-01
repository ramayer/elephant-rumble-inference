#!/usr/bin/env python

import torch
import argparse
import einops
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier
from .audio_file_processor import AudioFileProcessor
from .audio_file_visualizer import AudioFileVisualizer
from .raven_file_helper import RavenFileHelper

# consider: https://www.youtube.com/watch?v=Qw9TmrAIS6E for demos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO - try Apple MPS

print(f"using {DEVICE}")
if DEVICE=='cpu':
  print("WARNING - this can be extremely slow on the CPU - prepare to wait a long time")
  print("   Recommend running with --limit-audio-hours=1 ")

def parse_args():
    
    usage = r"""Usage:
        elephant-rumble-inference  /tmp/test.wav \
          --save-classification-scores=/tmp/scores.pt \
          --save-visualizations=/tmp/vis.png \
          --limit-audio-hours=1 \
          --discrete-colormap \
          --load-raven-files ~/proj/elephantlistening/data/Rumble 
    """
    parser = argparse.ArgumentParser(description="Find elephant rumbles in an audio clip",
                                     usage=usage)
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument("--save-visualizations", type=str, help="Save visualizations to a file (optional)")
    parser.add_argument("--save-classification-scores", type=str, help="Save classification scores to a file (optional)")
    parser.add_argument("--save-raven-file", type=str, help="Specify the name of the Raven file (optional)")
    parser.add_argument("--load-raven-files", type=str, help="for the visualizer")
    parser.add_argument("--discrete-colormap", action="store_true", help="Make output visualization use a discrete color map")
    parser.add_argument("--limit-audio-hours", type=int, default=24, help="Limit audio hours (default: 24)")
    args = parser.parse_args()
    return args

def initialize_models():
    atw = AvesTorchaudioWrapper().to(DEVICE)
    erc = ElephantRumbleClassifier().to("cpu")
    erc_weights = erc.choose_model_weights("best_using_training_data_only")
    erc.load_pretrained_weights(erc_weights)
    return atw, erc

def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Input files: {args.input_files}")
    atw, erc = initialize_models()
    afp = AudioFileProcessor(atw, erc,device=DEVICE)
    for audio_file in args.input_files:
        scores = afp.classify_wave_file_for_rumbles(audio_file,limit_audio_hours=args.limit_audio_hours)
        if args.save_classification_scores:
          torch.save(scores, args.save_classification_scores)
        if args.save_visualizations:
          if args.load_raven_files:
             rfh = RavenFileHelper(args.load_raven_files)
             lbls = rfh.get_all_labels_for_wav_file('CEB1_20111010_000000.wav')
          else:
             lbls = []

          AudioFileVisualizer().visualize_audio_file_fragment(
              f"{audio_file} and scores",
              args.save_visualizations,
              audio_file,
              scores[:,1],
              scores[:,0],
              afp,
              start_time=0,
              end_time=60*5,
              make_discrete=args.discrete_colormap,
              labels = lbls
          )

