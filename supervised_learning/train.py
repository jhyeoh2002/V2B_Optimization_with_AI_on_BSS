import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.temporal_attentive_fusion_net import TemporalAttentiveFusionNet
from data.loader import get_loaders
from utils import set_seed, save_metrics, plot_training_curves

def train_model(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(opts.dropcolumns, opts.batch_size, ...)

    model = TemporalAttentiveFusionNet(
        num_embeddings=opts.num_embeddings,
        embedding_dim=opts.embedding_dim,
        n_heads=opts.n_heads,
        ...
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

    best_loss = float('inf')
    for epoch in range(opts.epochs):
        # training loop
        # validation loop
        # logging
        # save model if improved
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropcolumns', nargs='+', default=[...])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--lr',         type=float, default=0.005)
    parser.add_argument('--num_embeddings', type=int, default=10000)
    parser.add_argument('--embedding_dim',   type=int, default=128)
    parser.add_argument('--n_heads',         type=int, default=4)
    parser.add_argument('--step_size',       type=int, default=1000)
    parser.add_argument('--gamma',           type=float, default=0.5)
    opts = parser.parse_args()
    train_model(opts)
