import torch
import argparse
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE}")


def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Input files: {args.input_files}")
    atw, erc = initialize_models()


def initialize_models():
    atw = AvesTorchaudioWrapper()
    erc = ElephantRumbleClassifier()
    erc_weights = erc.choose_model_weights("best_using_training_data_only")
    erc.load_pretrained_weights(erc_weights)
    print("Hello world!")
    return atw, erc


if __name__ == "__main__":
    main()
