import torch
import torch.nn as nn
import math

class TemporalAttentiveFusionNet(nn.Module):
    def __init__(self,
                 num_embeddings=10000,
                 embedding_dim=128,
                 n_heads=4,
                 fc_hidden_dim1=64,
                 fc_hidden_dim2=16,
                 attention_dropout=0.2,
                 dropout=0.2):
        """
        Interpretable temporal attention version:
        - Uses MultiheadAttention to model 24-hour temporal influence
        - Each of the 7 time-series (carbon, radiation, temp, etc.) gets an attention summary
        - Concatenated with static features and passed through FC layers
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = n_heads
        self.num_embeddings = num_embeddings

        # === Temporal embedding and attention ===
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.attn_fc = nn.Linear(embedding_dim, 1)  # summarize embedding to scalar per series

        # === Dropouts ===
        self.dropout = nn.Dropout(dropout)

        # === Fully connected fusion ===
        # 12 = number of static features + 7 temporal summaries (each -> 1 scalar)
        self.fc1 = nn.Linear(12, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, fc_hidden_dim2)
        self.fc4 = nn.Linear(fc_hidden_dim2, 1)

        # === Activation ===
        self.relu = nn.ReLU()

    def forward(self, x, sequence_length=24):
        # assume x shape = [batch, 2 + 5*sequence_length]
        static_feats = x[:, :2]  # first 2
        # now build the series blocks – assume series come in fixed order
        start = 2
        series1 = x[:, start : start + sequence_length]
        start += sequence_length
        series2 = x[:, start : start + sequence_length]
        start += sequence_length
        series3 = x[:, start : start + sequence_length]
        start += sequence_length
        series4 = x[:, start : start + sequence_length]
        start += sequence_length
        series5 = x[:, start : start + sequence_length]

        # get attention‐summaries
        s1 = self.temporal_attention(series1)
        s2 = self.temporal_attention(series2)
        s3 = self.temporal_attention(series3)
        s4 = self.temporal_attention(series4)
        s5 = self.temporal_attention(series5)

        # concatenate: static features + each summary scalar
        layer1 = torch.cat((static_feats, s1, s2, s3, s4, s5), dim=1)

        # then FC layers as before
        x = self.fc1(layer1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        out = self.fc4(x)
        return out


    def temporal_attention(self, seq):
        """
        seq: [batch, 24]
        Returns a scalar summary [batch, 1] after attention over 24 hours.
        """
        # Quantize continuous values for embedding lookup
        seq = ((seq * 1000).round() + 5000).long().clamp(0, self.num_embeddings - 1)
        emb = self.seriesEmbedding(seq)          # [B, 24, emb_dim]

        # Self-attention (each hour attends to every other)
        attn_out, attn_weights = self.attn(emb, emb, emb)  # [B, 24, emb_dim], [B, n_heads, 24, 24]
        attn_out = self.attn_norm(attn_out)

        # Mean pooling over time → one embedding per sequence
        pooled = attn_out.mean(dim=1)  # [B, emb_dim]

        # Optional: you can inspect attn_weights for interpretability per head/hour
        summary = self.attn_fc(pooled)  # [B, 1]
        return summary
