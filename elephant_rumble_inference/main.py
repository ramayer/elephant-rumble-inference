#!/usr/bin/env python

import torch
import argparse
import einops
from .aves_torchaudio_wrapper import AvesTorchaudioWrapper
from .elephant_rumble_classifier import ElephantRumbleClassifier

# consider: https://www.youtube.com/watch?v=Qw9TmrAIS6E for demos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE}")


def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model", type=str, help="Specify the model name")
    parser.add_argument("input_files", nargs="*", help="List of input files")
    parser.add_argument(
        "--save-classification-scores",
        action="store_true",
        help="Save classification scores (default: False)",
    )
    args = parser.parse_args()
    return args

    parser.add_argument("--save-classification-scores", type=str, help="save")


import torchaudio.io as tai


class AudioFileProcessor:
    def __init__(self, aves_model, elephant_model, rumble_sr=500):
        self.aves_model = aves_model
        self.elephant_model = elephant_model
        self.rumble_sr = rumble_sr

    def get_aves_embeddings(self, chunk):
        with torch.inference_mode():
            y32 = chunk.to(torch.float32).view(1, chunk.shape[0]).to(DEVICE)
            aves_embeddings = self.aves_model.forward(y32).to("cpu").detach()
            if torch.cuda.is_available():
                del y32  # free space on my small cheap GPU
                torch.cuda.empty_cache()
            reshaped_tensor = einops.rearrange(
                aves_embeddings, "1 n d -> n d"
            )  # remove that batch dimension
            return reshaped_tensor.to("cpu").detach()

    def classify_wave_file_for_rumbles(self, wav_file_path):
        streamer = tai.StreamReader(wav_file_path)
        streamer.add_basic_audio_stream(
            stream_index=0,
            sample_rate=self.rumble_sr,
            frames_per_chunk=self.rumble_sr * 60 * 60,
        )
        results = []
        for idx, (chunk,) in enumerate(streamer.stream()):
            if chunk is not None:
                with torch.inference_mode():  # torch.no_grad():
                    print(f"processing hour {idx} of  {wav_file_path}")
                    aves_embeddings = self.get_aves_embeddings(chunk)
                    rumble_classification = self.elephant_model.forward(aves_embeddings)
                    results.append(rumble_classification)
                    print(
                        f", input samples {chunk.shape}, embedding_samples = {aves_embeddings.shape}, predictions = {rumble_classification.shape}"
                    )
                    if idx > 1:  # for unit testing
                        break
        return torch.cat(results)


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
    afp = AudioFileProcessor(atw, erc)
    for f in args.input_files:
        scores = afp.classify_wave_file_for_rumbles(f)
        if args.save_classification_scores:
          torch.save(scores, f"{f}.rumble_classification_scores.pt")
