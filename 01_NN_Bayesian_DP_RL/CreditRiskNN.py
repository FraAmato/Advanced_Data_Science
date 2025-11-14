class RobustCreditRiskModel(nn.Module):
    def __init__(self, input_size=12, hidden_sizes=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Linear layer with proper initialization
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            
            layers.append(linear)
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch norm
            layers.append(nn.ReLU())                     # ReLU activation
            layers.append(nn.Dropout(0.2))               # Dropout for regularization
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training with all best practices
model = RobustCreditRiskModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
criterion = nn.BCELoss()

for epoch in range(epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch.features)
        loss = criterion(outputs, batch.labels)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Learning rate scheduling
    scheduler.step(validation_loss)

# Trasking gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f'Gradient norm: {total_norm}')

# Monitor dead neurons

def count_dead_neurons(model, data_loader):
    dead_count = 0
    total_count = 0
    
    with torch.no_grad():
        for data in data_loader:
            for module in model.modules():
                if isinstance(module, nn.ReLU):
                    activations = module(data)
                    dead_count += (activations == 0).sum().item()
                    total_count += activations.numel()
    
    return dead_count / total_count