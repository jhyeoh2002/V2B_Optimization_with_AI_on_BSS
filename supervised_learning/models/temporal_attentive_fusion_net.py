import torch
import torch.nn as nn
import math

class TemporalAttentiveFusionNet(nn.Module):
    def __init__(self,
                 num_embeddings=8000,
                 embedding_dim=64,
                 n_heads=4,
                 fc_hidden_dim1=128,
                 fc_hidden_dim2=64,
                 fc_hidden_dim3=16,
                 attention_dropout=0.4,
                 dropout=0.4):
        """
        Multihead attention-based fusion of static + temporal inputs.
        Each 24-hour series passes through interpretable attention, then all summaries are fused.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = n_heads
        self.num_embeddings = num_embeddings

        # === Temporal embedding + multihead attention ===
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.attn_fc = nn.Linear(embedding_dim, 1)  # summarises attention output

        # === Fully connected fusion ===
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(4 +(24*5) + 5, fc_hidden_dim1)  # 4 static + 5 temporal summaries
        self.bn1 = nn.BatchNorm1d(fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.bn2 = nn.BatchNorm1d(fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, fc_hidden_dim3)
        self.fc4 = nn.Linear(fc_hidden_dim3, 1)
        self.relu = nn.ReLU()

    def forward(self, x, sequence_length=24):
        """
        x shape: [batch_size, total_features]
        total_features = 4 static + (5 * 24 temporal series)
        """
        static_feats = x[:]  # sin/cos features
        temporal_feats = x[:, 4:].reshape(x.shape[0], 5, sequence_length)

        summaries = []
        for i in range(5):
            seq = temporal_feats[:, i, :]  # [B, 24]
            summaries.append(self.temporal_attention(seq))
        summaries = torch.cat(summaries, dim=1)  # [B, 5]

        # Fuse static + attention summaries
        fused = torch.cat([static_feats, summaries], dim=1)
       
        out = self.fc1(fused)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)

        return out

    def temporal_attention(self, seq):
        """
        seq: [batch, 24]
        Returns [batch, 1] summary after multihead attention
        """
        seq = ((seq * 1000).round() + 5000).long().clamp(0, self.num_embeddings - 1)
        emb = self.seriesEmbedding(seq)
        attn_out, _ = self.attn(emb, emb, emb)
        attn_out = self.attn_norm(attn_out)
        pooled = attn_out.mean(dim=1)
        return self.attn_fc(pooled)  # [B, 1]
