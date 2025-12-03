import torch
import torch.nn as nn
import math

class TemporalAttentiveFusionNet(nn.Module):
    def __init__(self,
                 num_embeddings=10000,
                 embedding_dim=64,
                 n_heads=4,
                 fc_hidden_dim1=128,
                 fc_hidden_dim2=16,
                 dropout=0.3,
                 attention_dropout=0.3):
        """
        Complete V2B Model with Temporal Resolution Preservation.
        """
        super().__init__()

        # ======================================================================
        # 1. TEMPORAL ATTENTION BLOCK (Shared Weights)
        # ======================================================================
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Embeds discrete/quantized values into vectors
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Multi-Head Attention to find relationships between time steps
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embedding_dim)
        
        # Projector: Maps the 64-dim embedding back to a scalar importance value
        # We map to 1 so we get [Batch, 24, 1] -> squeeze -> [Batch, 24]
        self.attn_fc = nn.Linear(embedding_dim, 1)


        # ======================================================================
        # 2. FUSION & MLP BLOCK
        # ======================================================================
        # CALCULATION OF INPUT DIMENSIONS:
        # A. Static Global (Time):              4 features
        # B. Static Vehicles (76 cars * 1):   76 features
        # C. Attended Temporal (5 series * 24): 120 features (Output of Attention)
        # -----------------------------------------------------
        # TOTAL FUSION INPUT:                 200 features
        # ======================================================================
        fusion_input_dim = 4 + 76 + 120
        
        # Layer 1
        self.bn0 = nn.BatchNorm1d(fusion_input_dim)
        self.fc1 = nn.Linear(fusion_input_dim, fc_hidden_dim1)
        self.bn1 = nn.BatchNorm1d(fc_hidden_dim1)
        
        # Layer 2
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.bn2 = nn.BatchNorm1d(fc_hidden_dim2)
        
        # Output Layer (Regression)
        self.fc4 = nn.Linear(fc_hidden_dim2, 1)
        
        # Activations
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        x shape: [batch_size, 428]
        """
        
        # --- STEP 1: SLICE THE INPUT DATA ---
        
        # 1. Global Time (Indices 0, 1, 2, 3)
        global_time = x[:, 0:4] 
        
        # 2. Environmental Series (Indices 4 to 124) -> Total 120
        # Reshape to: [Batch, 5 series, 24 hours]
        # The 5 series are: Building Load, Radiation, Temp, Price, Battery(Station)
        temporal_feats = x[:, 4:124].reshape(x.shape[0], 5, 24)

        # 3. Vehicle Fleet Data (Indices 124 to end) -> Total 304
        # (76 vehicles * 4 vars: SOC, DepSOC, DepTime, Priority)
        vehicle_data = x[:, 124:]


        # --- STEP 2: TEMPORAL ATTENTION ---
        # We process each of the 5 series through the attention mechanism
        summaries = []
        for i in range(5):
            seq = temporal_feats[:, i, :]  # Extract one 24h series: [Batch, 24]
            
            # Output is [Batch, 24] (refined curve)
            processed_seq = self.temporal_attention(seq)
            summaries.append(processed_seq)
            
        # Concatenate the 5 processed series -> [Batch, 120]
        # We flatten them to feed into the MLP
        attn_output = torch.cat(summaries, dim=1) 


        # --- STEP 3: FUSION ---
        # Combine: Global Time (4) + Vehicles (304) + Attended Env (120)
        fused = torch.cat([global_time, vehicle_data, attn_output], dim=1)


        # --- STEP 4: MLP PREDICTION ---
        out = self.bn0(fused)

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.fc4(out) # Final Prediction

        return out


    def temporal_attention(self, seq):
        """
        Applies Self-Attention to a 24-hour sequence.
        Input:  [Batch, 24] (Raw values)
        Output: [Batch, 24] (Context-aware values)
        """
        # 1. Quantization & Embedding
        # Note: Ensure your data scaling fits within (0, num_embeddings-1)
        # Assuming input is normalized roughly around 0-1 or similar before this formula
        seq_idx = ((seq * 1000).round() + 5000).long()
        seq_idx = seq_idx.clamp(0, self.num_embeddings - 1)
        
        emb = self.seriesEmbedding(seq_idx) # [Batch, 24, 64]

        # 2. Self Attention
        # Q, K, V are all the same (Self-Attention)
        # attn_out: [Batch, 24, 64]
        attn_out, _ = self.attn(emb, emb, emb)
        
        # 3. Residual Connection & Norm (Optional but recommended, here just Norm)
        # You could also do: attn_out = attn_out + emb
        attn_out = self.attn_norm(attn_out)

        # 4. Project back to scalar per timestep
        # [Batch, 24, 64] -> Linear -> [Batch, 24, 1]
        out = self.attn_fc(attn_out) 
        
        # 5. Remove last dimension to get [Batch, 24]
        return out.squeeze(-1)

# ==========================================
# SANITY CHECK (To ensure no shape errors)
# ==========================================
if __name__ == "__main__":
    # Create model
    model = TemporalAttentiveFusionNet()
    
    # Create dummy input: Batch size 32, 428 features
    dummy_input = torch.randn(32, 428)
    
    # Forward pass
    try:
        output = model(dummy_input)
        print("✅ Model instantiation successful.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # Should be [32, 1]
    except Exception as e:
        print(f"❌ Error: {e}")