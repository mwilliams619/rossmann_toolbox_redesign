class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.num_epochs = 100
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_clip_val = 1.0
        self.early_stopping_patience = 10

def train_epoch(model, dataloader, criterion, optimizer, config):
    """Single training epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        embeddings = batch['embeddings'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, config):
    """Validation step"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)