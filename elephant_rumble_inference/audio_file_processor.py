
import einops
import torch
import torchaudio.io as tai
from .triple_buffered_iterator import TripleBufferedIterator

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
        for idx, (prv,cur,nxt) in enumerate(TripleBufferedIterator(streamer.stream())):
            (chunk,) = cur
            if chunk is not None:

                # Note - if an hour at a 512Hz framerate has 1800000, samples 
                # we expect 5625 AVES/Hubert embeddings, each representing 0.625 seconds.

                with torch.inference_mode():  # torch.no_grad():
                    if chunk.shape[0] % 320 != 0:
                        print("""
                              Warning - AVES/Hubert uses 320 sample convolutional layers; 
                              the last embedding vector may be based on incomplete information.
                              """)
                    preroll = torch.empty(0,1)
                    postroll = torch.empty(0,1)
                    if nxt is not None:
                        postroll = nxt[0][0:320*16] # 8 is not enough
                    if prv is not None and prv[0].shape[0] >= 320*16:
                        preroll = prv[0][-320*16:]
                    print(f"Classifying hour {idx} of {wav_file_path} {preroll.shape}, {chunk.shape}, {postroll.shape}")

                    chunk_for_aves = torch.concat([preroll,chunk,postroll])
                    aves_embeddings = self.get_aves_embeddings(chunk_for_aves)
                    aves_embeddings = self.normalize_aves_embeddings(aves_embeddings) # to compare with cosine similiary
                    rumble_classification = self.elephant_model.forward(aves_embeddings)
                    #print(idx,"rumble classification shape",rumble_classification.shape)

                    end_of_preroll = preroll.shape[0] // 320
                    beg_of_postroll = (preroll.shape[0] + chunk.shape[0])//320 # + 1

                    #print("Trimming",end_of_preroll, beg_of_postroll)
                    rumble_classification = rumble_classification[end_of_preroll:beg_of_postroll]
                    #print("###########",rumble_classification.shape)

                    results.append(rumble_classification)
                    if nxt is None:
                        # if there's no postroll, AVES doesn't return the last sample.
                        results.append(rumble_classification[-2:-1,:])
                    if idx+1 >= limit_audio_hours:  # for unit testing
                        break

        if len(results) == 0:
            print(f"Warning - two few audio samples to classify in {wav_file_path}")
            return torch.empty(0,768)
        return torch.cat(results)
