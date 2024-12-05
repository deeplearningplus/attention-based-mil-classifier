from model import *

# Example dataset (dummy data)
batch_size = 16
num_instances = 10  # Number of instances per bag
input_dim = 512
hidden_dim = 384
attention_dim = 256
X_train = torch.randn(batch_size, num_instances, input_dim)  # (batch_size, num_instances, input_dim)
y_train = torch.randint(0, num_classes, (batch_size,))  # (batch_size,)

net = AttentionMILClassifier(input_dim, hidden_dim, attention_dim, num_classes)
print(net)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    net.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs, attention_weights = net(X_train)
    
    # Compute the loss
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Inference to get attention weights
net.eval()
with torch.no_grad():
    _, attention_weights = net(X_train)

# Attention weights for each sample
print(attention_weights.squeeze().numpy())  # shape: (batch_size, num_instances, 1)
