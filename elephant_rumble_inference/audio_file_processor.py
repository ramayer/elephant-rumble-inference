
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

    def normalize_aves_embeddings(self,embs):
        with torch.inference_mode(): # torch.no_grad():
            norms = embs.norm(p=2, dim=1, keepdim=True)
            unit_vecs = embs / norms
            return unit_vecs.to('cpu').detach()

    def get_aves_embeddings(self, chunk):
        with torch.inference_mode():
            chunk = chunk[:,0:1]  # remove stereo or surround channels
            #print("in get_aves_embeddngs",chunk.shape)
            if chunk.shape[0] < 320*2:
                print("Warning - two few audio samples to classify in chunk")
                return torch.empty(0, 768)
            y32 = chunk.to(torch.float32).view(1, chunk.shape[0]).to(self.device)
            aves_embeddings = self.aves_model.forward(y32).to("cpu").detach()
            if torch.cuda.is_available():
                del y32  # free space on my small cheap GPU
                torch.cuda.empty_cache()
            reshaped_tensor = einops.rearrange(
                aves_embeddings, "1 n d -> n d"
            )  # remove that batch dimension
            #print("reshaped tensor shape is",reshaped_tensor.shape)
            return reshaped_tensor.to("cpu").detach()

    def classify_wave_file_for_rumbles(self, wav_file_path, limit_audio_hours=24 ):
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
                    print(f"Classifying hour {idx} of {wav_file_path}")
                    aves_embeddings = self.get_aves_embeddings(chunk)
                    aves_embeddings = self.normalize_aves_embeddings(aves_embeddings) # to compare with cosine similiary
                    rumble_classification = self.elephant_model.forward(aves_embeddings)
                    ##print("ERROR!!!  {rumble_classification.shape} does not seem to equal an hour",
                    ##      "Need to pad or interpolate or replay fragments of the previous hour???",
                    ##      "Or pick a multiple of 320 for the frames per chunk?",
                    ##      )
                    ## TODO: Better to save the final 320 samples from the previous frame
                    ## and prepend it to the new frame.
                    ##
                    ##  Even better -- save many samples from the previous frame to re-set the state of the
                    ##  transformer's attention blocks that look back in time.
                    ##
                    results.append(rumble_classification)
                    results.append(rumble_classification[-2:-1,:])
                    if idx+1 >= limit_audio_hours:  # for unit testing
                        break
        if len(results) == 0:
            print(f"Warning - two few audio samples to classify in {wav_file_path}")
            return torch.empty(0,768)
        return torch.cat(results)
