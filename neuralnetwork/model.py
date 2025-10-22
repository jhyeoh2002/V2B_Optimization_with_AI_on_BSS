import torch
import torch.nn as nn
import math

class Integrated_Model(nn.Module):
    
    def __init__(self,
                num_embeddings = 5000, 
                embedding_dim = 256,
                fc_hidden_dim1 = 32,
                fc_hidden_dim2 = 8,
                attention_dropout = 0.1,
                dropout = 0.2
                ):
        
        super(Integrated_Model, self).__init__()
        
        self.embedding_dim = embedding_dim 
        
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        self.attention = nn.Linear(embedding_dim, embedding_dim)
        self.seqoutput = nn.Linear(embedding_dim, 1)
        self.seqoutput2 = nn.Linear(24, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(12, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, 1)
        
    def forward(self, x, sequence_length = 24):
        
        features = (x[:,:-sequence_length*7])
        carbInt = self.selfattention(x[:,-sequence_length*7:-sequence_length*6])
        radiation = self.selfattention(x[:,-sequence_length*6:-sequence_length*5])
        temp = self.selfattention(x[:,-sequence_length*5:-sequence_length*4])
        pvgen = self.selfattention(x[:,-sequence_length*4:-sequence_length*3])
        bdg_elec = self.selfattention(x[:,-sequence_length*3:-sequence_length*2])
        vehc_elec = self.selfattention(x[:,-sequence_length*2:-sequence_length])
        energy = self.selfattention(x[:,-sequence_length:])
        # print("Shape of features:", features.shape)  # Print output shape    

        layer1 = torch.cat((features, carbInt, radiation, temp, pvgen, bdg_elec, vehc_elec, energy), dim=1)
        
        # Fully Connected Layers
        fc_output = self.fc1(layer1)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        fc_output = self.fc2(fc_output)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        final_output = self.fc3(fc_output)
        # print("Final output size:", final_output.size())

        return final_output

        
    def selfattention(self, x):
        x = ((x*1000).round() + 2500).long()  # Round, shift to positive range, and convert to LongTensor

        # Clamp indices to valid range [0, num_embeddings - 1]
        x = x.clamp(0, self.seriesEmbedding.num_embeddings - 1)
        embedded_sequence = self.seriesEmbedding(x)
        
        query = self.attention(embedded_sequence)
        key = self.attention(embedded_sequence)
        value = self.attention(embedded_sequence)
        
        matmul_qk = query @ key.transpose(-2, -1)
        scaled_matmul_qk = matmul_qk / math.sqrt(self.embedding_dim)
        
        attention_weights = self.softmax(scaled_matmul_qk)
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = attention_weights @ value
        attention_output = self.attention_dropout(attention_output)

        output = self.seqoutput(attention_output).squeeze(-1)
        final_output = self.seqoutput2(output)

        # print("Shape of output:", output.size())  # Print output shape    
           
        return final_output
    
    
class Integrated_Model2(nn.Module):
    
    def __init__(self,
                num_embeddings = 10000, 
                embedding_dim = 256,
                fc_hidden_dim1 = 32,
                fc_hidden_dim2 = 8,
                attention_dropout = 0.4,
                dropout = 0.2
                ):
        
        super(Integrated_Model2, self).__init__()
        
        self.embedding_dim = embedding_dim 
        
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        self.attention = nn.Linear(embedding_dim, embedding_dim)
        self.seqoutput = nn.Linear(embedding_dim, 1)
        self.seqoutput2 = nn.Linear(24, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(12, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, fc_hidden_dim2)
        self.fc4 = nn.Linear(fc_hidden_dim2, 1)
        
    def forward(self, x, sequence_length = 24):
        
        features = (x[:,:-sequence_length*7])
        carbInt = self.selfattention(x[:,-sequence_length*7:-sequence_length*6])
        radiation = self.selfattention(x[:,-sequence_length*6:-sequence_length*5])
        temp = self.selfattention(x[:,-sequence_length*5:-sequence_length*4])
        pvgen = self.selfattention(x[:,-sequence_length*4:-sequence_length*3])
        bdg_elec = self.selfattention(x[:,-sequence_length*3:-sequence_length*2])
        vehc_elec = self.selfattention(x[:,-sequence_length*2:-sequence_length])
        energy = self.selfattention(x[:,-sequence_length:])
        # print("Shape of features:", features.shape)  # Print output shape    

        layer1 = torch.cat((features, carbInt, radiation, temp, pvgen, bdg_elec, vehc_elec, energy), dim=1)
        
        # Fully Connected Layers
        fc_output = self.fc1(layer1)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        fc_output = self.fc2(fc_output)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        fc_output = self.fc3(fc_output)
        fc_output = nn.ReLU()(fc_output)
        
        final_output = self.fc4(fc_output)
        # print("Final output size:", final_output.size())

        return final_output

        
    def selfattention(self, x):
        x = ((x*1000).round() + 5000).long()  # Round, shift to positive range, and convert to LongTensor

        # Clamp indices to valid range [0, num_embeddings - 1]
        x = x.clamp(0, self.seriesEmbedding.num_embeddings - 1)
        embedded_sequence = self.seriesEmbedding(x)
        
        query = self.attention(embedded_sequence)
        key = self.attention(embedded_sequence)
        value = self.attention(embedded_sequence)
        
        matmul_qk = query @ key.transpose(-2, -1)
        scaled_matmul_qk = matmul_qk / math.sqrt(self.embedding_dim)
        
        attention_weights = self.softmax(scaled_matmul_qk)
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = attention_weights @ value
        attention_output = self.attention_dropout(attention_output)

        output = self.seqoutput(attention_output).squeeze(-1)
        final_output = self.seqoutput2(output)

        # print("Shape of output:", output.size())  # Print output shape    
           
        return final_output
    
class Integrated_Model3(nn.Module):
    
    def __init__(self,
                num_embeddings = 10000, 
                embedding_dim = 512,
                fc_hidden_dim1 = 64,
                fc_hidden_dim2 = 16,
                attention_dropout = 0.2,
                dropout = 0.2
                ):
        
        super(Integrated_Model3, self).__init__()
        
        self.embedding_dim = embedding_dim 
        
        self.seriesEmbedding = nn.Embedding(num_embeddings, embedding_dim)
        self.attention = nn.Linear(embedding_dim, embedding_dim)
        self.seqoutput = nn.Linear(embedding_dim, 1)
        self.seqoutput2 = nn.Linear(24, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(12, fc_hidden_dim1)
        self.fc2 = nn.Linear(fc_hidden_dim1, fc_hidden_dim2)
        self.fc3 = nn.Linear(fc_hidden_dim2, fc_hidden_dim2)
        self.fc4 = nn.Linear(fc_hidden_dim2, 1)
        
    def forward(self, x, sequence_length = 24):
        
        features = (x[:,:-sequence_length*7])
        carbInt = self.selfattention(x[:,-sequence_length*7:-sequence_length*6])
        radiation = self.selfattention(x[:,-sequence_length*6:-sequence_length*5])
        temp = self.selfattention(x[:,-sequence_length*5:-sequence_length*4])
        pvgen = self.selfattention(x[:,-sequence_length*4:-sequence_length*3])
        bdg_elec = self.selfattention(x[:,-sequence_length*3:-sequence_length*2])
        vehc_elec = self.selfattention(x[:,-sequence_length*2:-sequence_length])
        energy = self.selfattention(x[:,-sequence_length:])
        # print("Shape of features:", features.shape)  # Print output shape    

        layer1 = torch.cat((features, carbInt, radiation, temp, pvgen, bdg_elec, vehc_elec, energy), dim=1)
        
        # Fully Connected Layers
        fc_output = self.fc1(layer1)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        fc_output = self.fc2(fc_output)
        fc_output = nn.ReLU()(fc_output)
        fc_output = self.dropout(fc_output)

        fc_output = self.fc3(fc_output)
        fc_output = nn.ReLU()(fc_output)
        
        final_output = self.fc4(fc_output)
        # print("Final output size:", final_output.size())

        return final_output

        
    def selfattention(self, x):
        x = ((x*1000).round() + 5000).long()  # Round, shift to positive range, and convert to LongTensor

        # Clamp indices to valid range [0, num_embeddings - 1]
        x = x.clamp(0, self.seriesEmbedding.num_embeddings - 1)
        embedded_sequence = self.seriesEmbedding(x)
        
        query = self.attention(embedded_sequence)
        key = self.attention(embedded_sequence)
        value = self.attention(embedded_sequence)
        
        matmul_qk = query @ key.transpose(-2, -1)
        scaled_matmul_qk = matmul_qk / math.sqrt(self.embedding_dim)
        
        attention_weights = self.softmax(scaled_matmul_qk)
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = attention_weights @ value
        attention_output = self.attention_dropout(attention_output)

        output = self.seqoutput(attention_output).squeeze(-1)
        final_output = self.seqoutput2(output)

        # print("Shape of output:", output.size())  # Print output shape    
           
        return final_output