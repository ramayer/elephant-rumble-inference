import json
import torch
from torchaudio.models import wav2vec2_model

class AvesTorchaudioWrapper(torch.nn.Module):

    def __init__(self, 
                 config_path='aves-base-bio.torchaudio.model_config.json', 
                 model_path='aves-base-bio.torchaudio.pt'):
        super().__init__()
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def forward(self, sig):
        out = self.model.extract_features(sig)[0][-1]
        return out
