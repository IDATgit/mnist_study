import torch
import sys
import os

# Current file path and model_name
current_filename = os.path.basename(__file__)
model_name = current_filename[:-3]

# Fix import path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from models.specific_models.GrokkingTinyTransformer import GrokkingTinyTransformer
from trainers.generic_trainers.basic_trainer import BasicTrainer
from utils.data_loader import ModularArithmeticDataLoader

# Modular arithmetic config (matches grokking recipe)
P = 97
OP = '/'
TRAIN_FRACTION = 0.5
SEQ_LEN = 4  # [a, op, b, =]
VOCAB_SIZE = P + 2  # residues + op + '='

# Model
# def __init__(self, depth, dim, heads, n_tokens, seq_len, dropout=0., pool='cls'):
model = GrokkingTinyTransformer(
    depth=2,
    dim=128,
    heads=1,
    n_tokens=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    dropout=0.0,
    pool='cls',
)
print("Number of parameters: ", model.get_num_parameters())

# Data
data_loader = ModularArithmeticDataLoader(
    p=P,
    op=OP,
    train_fraction=TRAIN_FRACTION,
    batch_size=512,
    seed=42,
)


def train_grokking_modular_division_p97():
    trainer = BasicTrainer(
        model=model,
        model_name=model_name,
        learning_rate=3e-4,
        num_epochs=350,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        data_loader=data_loader,
        visualization=False
    )
    # Align optimizer with grokking defaults: AdamW(lr=1e-3, weight_decay=1, betas=(0.9, 0.98))
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
    trainer.train()
    return trainer.get_history()


if __name__ == "__main__":
    train_grokking_modular_division_p97()

