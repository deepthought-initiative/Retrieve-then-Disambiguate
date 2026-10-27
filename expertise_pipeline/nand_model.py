"""The NAND encoder: maps an 818-dim (chars2vec + SPECTER2) mention vector to a 256-dim embedding
that dbscanc.py clusters. Architecture matches AND_nn_exp.py's ANDismabiguator.l1 exactly (same
layer names/order), so a checkpoint trained with that pytorch_lightning module loads directly here
via load_state_dict -- this is a plain nn.Module wrapping just the encoder, not the full
LightningModule with its training/loss code, since only inference is needed downstream.

https://github.com/deepthought-initiative/neural_name_dismabiguator
https://iopscience.iop.org/article/10.1088/1538-3873/ae1e2d
"""
import numpy as np
import torch
from torch import nn

CHECKPOINT_PATH = '1_specter_chars2vec_nceepoch=67-val_loss=7.47.ckpt'

INPUT_DIM = 818   # 50 (chars2vec) + 768 (SPECTER2)
OUTPUT_DIM = 256


class NANDEncoder(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(INPUT_DIM, 1024),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, OUTPUT_DIM),
            nn.SELU(),
        )

    def forward(self, x):
        return self.l1(x)


def load_nand_model(checkpoint_path=CHECKPOINT_PATH, device='cpu'):
    """Loads weights from a pytorch_lightning checkpoint (state_dict keys are 'l1.0.weight', etc.,
    matching this module's own 'l1' attribute directly -- no key remapping needed)."""
    model = NANDEncoder()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_mentions(model, chars2vec_embeddings, specter_embeddings, device='cpu'):
    """chars2vec_embeddings: (n, 50), specter_embeddings: (n, 768) -> (n, 256) numpy array."""
    x = torch.tensor(
        np.concatenate([chars2vec_embeddings, specter_embeddings], axis=1), dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()
