import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(AttentionMIL, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_U = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, num_instances, input_dim)
        H = F.relu(self.fc1(x))  # shape: (batch_size, num_instances, hidden_dim)
        
        # Calculate attention weights
        A_V = torch.tanh(self.attention_V(H))  # shape: (batch_size, num_instances, attention_dim)
        A_U = torch.sigmoid(self.attention_U(H))  # shape: (batch_size, num_instances, attention_dim)
        A = A_V * A_U  # Gated attention
        attention_weights = F.softmax(self.attention_w(A), dim=1)  # shape: (batch_size, num_instances, 1)
        
        # Compute bag representation as a weighted sum of instance embeddings
        z = torch.sum(attention_weights * H, dim=1)  # shape: (batch_size, hidden_dim)
        
        return z, attention_weights

class AttentionMILClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, num_classes):
        super(AttentionMILClassifier, self).__init__()
        self.attention_mil = AttentionMIL(input_dim, hidden_dim, attention_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        z, attention_weights = self.attention_mil(x)  # z is the bag representation
        output = self.classifier(z)  # shape: (batch_size, num_classes)
        return output, attention_weights
