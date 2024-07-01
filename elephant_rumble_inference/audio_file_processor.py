
import einops
import torch
import torchaudio.io as tai

class AudioFileProcessor:
    def __init__(self, aves_model, elephant_model, rumble_sr=500, device="cuda"):
        self.aves_model = aves_model
        self.elephant_model = elephant_model
        self.rumble_sr = rumble_sr
        self.device = device
        self.audio_samples_per_embedding = 320 # from https://arxiv.org/pdf/2106.07447

    def time_to_score_index(self,t):
        return t * self.rumble_sr // self.audio_samples_per_embedding

    def score_index_to_time(self,s):
        return s * self.audio_samples_per_embedding / self.rumble_sr

    def get_aves_embeddings(self, chunk):
        with torch.inference_mode():
            y32 = chunk.to(torch.float32).view(1, chunk.shape[0]).to(self.device)
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
                    if idx > 1:  # for unit testing
                        break
        return torch.cat(results)
