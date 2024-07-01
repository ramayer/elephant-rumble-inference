import torch.nn as nn
import torch
import os

class ElephantRumbleClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=768 // 4,
        output_dim=2,
        dropout=0.2,
    ):
        super(ElephantRumbleClassifier, self).__init__()
        self.act = nn.LeakyReLU()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def get_cache_prefix(self):
        cache_prefix = torch.hub._get_torch_home()
        cache_dir = os.path.join(cache_prefix, "fruitpunch_elephants")
        return cache_dir

    def choose_model_weights(self, criteria):
        if criteria in ["training", "best_using_training_data_only"]:
            return "elephant_rumble_classifier_500_192_2024-06-29T23:39:01.415771_valloss=5.83.pth"
        if criteria in ["enhanced", "best_using_more_varied_training_data"]:
            return "elephant_rumble_classifier_500_192_2024-06-30T02:22:33.598037_valloss=6.55.pth"
        raise Exception("")

    def load_pretrained_weights(self, pretrained_weights):
        if not pretrained_weights.endswith(".pth"):
            pretrained_weights = self.choose_model_weights(pretrained_weights)
        self.download_model_files_if_needed(pretrained_weights)
        cache_dir  = self.get_cache_prefix()
        model_weights_file = os.path.join(cache_dir, pretrained_weights)
        self.load_state_dict(torch.load(model_weights_file))
        self.eval()

    def download_model_files_if_needed(self, pretrained_weights):
        src_prefix = "https://0ape.com/pretrained_models"
        dst_prefix = self.get_cache_prefix()
        src = f"{src_prefix}/{pretrained_weights}"
        dst = os.path.join(dst_prefix, pretrained_weights)
        if not os.path.exists(dst):
            print(f"fetching {dst} from {src}")
            os.makedirs(dst_prefix, exist_ok=True)
            torch.hub.download_url_to_file(src, dst)



