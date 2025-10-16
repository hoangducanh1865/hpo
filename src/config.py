

class Config:
    # Data
    batch_size = 256
    num_workers = 2
    
    # Model
    num_outputs = 10  # 10 classes for Fashion-MNIST
    
    # Training
    lr = 0.1
    num_epochs = 10
    device = "cpu" 
