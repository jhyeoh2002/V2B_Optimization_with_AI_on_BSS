import torch
import torch.nn as nn

class TemporalAttentiveFusionNet(nn.Module):
    """
    STAF-Net V3: Temporal Attentive Fusion Network.

    This model processes a flattened feature vector containing:
      1. Static Global Features (Time/Day)
      2. Multiple Time-Series (Building, Solar, Price, etc.)
      3. Vehicle/Battery Specific Features

    Architecture:
      - The time-series inputs pass through a shared Multi-Head Self-Attention mechanism
        to capture temporal dependencies.
      - The attention outputs are flattened and fused with Static and Vehicle features.
      - A Multi-Layer Perceptron (MLP) regresses the final target.
    """

    def __init__(self,
                 # Data Dimensions
                 num_static=4,
                 num_series=6,
                 sequence_length=24,
                 vehicle_input_dim=304, # (76 cars * 4 features) or similar
                 
                 # Model Hyperparameters
                 num_embeddings=10000,
                 embedding_dim=64,
                 n_heads=4,
                 fc_hidden_dim1=128,
                 fc_hidden_dim2=16,
                 fc_hidden_dim3=1024,
                 dropout=0.3,
                 attention_dropout=0.3):
        """
        Args:
            num_static (int): Number of static features at start of input vector.
            num_series (int): Number of distinct time-series (e.g., Load, Price, Temp).
            sequence_length (int): Length of each time-series window.
            vehicle_input_dim (int): Total count of remaining vehicle/battery features.
            num_embeddings (int): Vocabulary size for quantization embedding.
            embedding_dim (int): Dimension of the embedding space.
            n_heads (int): Number of attention heads.
            fc_hidden_dim1 (int): Neurons in first dense layer.
            fc_hidden_dim2 (int): Neurons in second dense layer.
            dropout (float): Dropout rate for MLP.
            attention_dropout (float): Dropout rate for Attention.
        """
        super().__init__()

        # --- Dimensions ---
        self.num_static = num_static
        self.num_series = num_series
        self.seq_len = sequence_length
        self.vehicle_dim = vehicle_input_dim
        self.num_embeddings = num_embeddings

        # Calculate indices for slicing the flattened input
        self.series_start = num_static
        self.series_end = num_static + (num_series * sequence_length)
        
        # --- 1. Temporal Attention Block (Shared Weights) ---
        # Embeds discrete/quantized values into vectors
        self.series_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.series_projection = nn.Linear(1, embedding_dim)
        
        # Multi-Head Attention to find relationships between time steps
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)
        
        # Projector: Maps the [Batch, Seq, Emb] -> [Batch, Seq, 1]
        self.attn_fc = nn.Linear(embedding_dim, 1)

        # --- 2. Fusion & MLP Block ---
        # Calculate fusion input size:
        # Static + Vehicle + (Num_Series * Seq_Len output from attention)
        # Note: In V3, attention preserves temporal resolution (Output is Seq_Len), 
        # so we concatenate 5 series * 24 hours = 120 features.
        fusion_input_dim = num_static + vehicle_input_dim + (num_series * sequence_length)
        
        self.mlp = nn.Sequential(
            # Layer 1
            nn.BatchNorm1d(fusion_input_dim),
            nn.Linear(fusion_input_dim, fc_hidden_dim1),
            nn.BatchNorm1d(fc_hidden_dim1),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(fc_hidden_dim1, fc_hidden_dim2),
            nn.BatchNorm1d(fc_hidden_dim2),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(fc_hidden_dim2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Flattened input [Batch_Size, Total_Features]
                              Total = Static + (Series * Seq) + Vehicle
        """
        batch_size = x.shape[0]

        # --- Step 1: Slice the Input ---
        # 1. Global Time 
        # Shape: [Batch, num_static]
        static_feats = x[:, 0 : self.series_start]

        # 2. Environmental Series
        # Raw Slice: [Batch, num_series * seq_len]
        # Reshape to: [Batch, num_series, seq_len]
        series_flat = x[:, self.series_start : self.series_end]
        series_feats = series_flat.view(batch_size, self.num_series, self.seq_len)

        # 3. Vehicle Data
        # Shape: [Batch, vehicle_dim]
        vehicle_feats = x[:, self.series_end :]

        # --- Step 2: Temporal Attention ---
        # We process each series (Load, Price, etc.) independently through the SHARED attention mechanism.
        
        attn_outputs = []
        for i in range(self.num_series):
            # Extract one series: [Batch, Seq_Len]
            seq = series_feats[:, i, :]
            
            # Apply Attention: Returns [Batch, Seq_Len]
            # This represents the "context-aware" version of the time series
            processed_seq = self._temporal_attention(seq)
            attn_outputs.append(processed_seq)
            
        # Concatenate processed series back into a flat vector
        # List of 5 tensors of [Batch, 24] -> Single tensor [Batch, 120]
        attn_flat = torch.cat(attn_outputs, dim=1) 

        # --- Step 3: Fusion ---
        # Combine: Static (4) + Vehicles (304) + Attended Series (120)
        # Shape: [Batch, 428]
        fused = torch.cat([static_feats, vehicle_feats, attn_flat], dim=1)

        # --- Step 4: Prediction ---
        return self.mlp(fused)

    def _temporal_attention(self, seq):
        """
        Internal helper: Applies Quantization -> Embedding -> Self-Attention.
        
        Args:
            seq (torch.Tensor): [Batch, Seq_Len] (Raw Float Values)
            
        Returns:
            torch.Tensor: [Batch, Seq_Len] (Refined Scalar Values)
        """
        #         # 1. On-the-fly Quantization (Float -> Long Index)
        #         # (val * 1000) + 5000 maps typical normalized values to positive integers
        #         seq_idx = ((seq * 1000).round() + 5000).long()
        #         seq_idx = seq_idx.clamp(0, self.num_embeddings - 1)
                
        #         # 2. Embedding
        #         # [Batch, Seq] -> [Batch, Seq, Emb_Dim]
        #         emb = self.series_embedding(seq_idx) 
        # def _temporal_attention(self, seq):
        """
        seq shape: [Batch, Seq_Len] (Float values, already scaled)
        """
        # 1. Unsqueeze to get feature dim: [Batch, Seq, 1]
        seq_vector = seq.unsqueeze(-1)
        
        # 2. Linear Projection: [Batch, Seq, 1] -> [Batch, Seq, Emb_Dim]
        emb = self.series_projection(seq_vector)
        
        # 3. Apply Attention (No change needed here)
        attn_out, _ = self.attn(emb, emb, emb)
        # 3. Self-Attention
        # Q, K, V are all 'emb'
        # attn_out: [Batch, Seq, Emb_Dim]
        # attn_out, _ = self.attn(emb, emb, emb)
        
        # 4. Residual/Norm (Standard Transformer practice)
        attn_out = self.attn_norm(attn_out + emb) # Added residual connection for stability

        # 5. Project back to scalar
        # [Batch, Seq, Emb] -> [Batch, Seq, 1]
        out = self.attn_fc(attn_out) 
        
        # 6. Squeeze last dim -> [Batch, Seq]
        return out.squeeze(-1)

# ==========================================
# Sanity Check
# ==========================================
if __name__ == "__main__":
    # Test parameters matching your dataset
    model = TemporalAttentiveFusionNet(
        num_static=4,
        num_series=5,
        sequence_length=24,
        vehicle_input_dim=304
    )
    
    # Create dummy input (Batch=32, Features=428)
    dummy_input = torch.randn(32, 428)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")   # [32, 428]
    print(f"Output shape: {output.shape}")       # [32, 1]
    print("✅ STAF-Net V3 Refactor Successful")