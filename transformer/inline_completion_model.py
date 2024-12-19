import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# 1. First, let's create a simple dataset class
class CodeCompletionDataset(Dataset):
    def __init__(self, input_lines: List[str], target_lines: List[str], tokenizer):
        self.input_lines = input_lines
        self.target_lines = target_lines
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.input_lines)
    
    def __getitem__(self, idx):
        # Tokenize input and target
        input_ids = self.tokenizer.encode(self.input_lines[idx], return_tensors='pt').squeeze(0)
        target_ids = self.tokenizer.encode(self.target_lines[idx], return_tensors='pt').squeeze(0)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

# 2. Create the model architecture
class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_encoder_layers: int = 6, max_length: int = 512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Embedding(max_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Create position indices
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        
        # Combine token embeddings and position encodings
        x = self.embedding(x) + self.position_encoding(positions)
        
        # Pass through transformer
        x = x.permute(1, 0, 2)  # Transform to sequence first format
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Transform back
        
        # Generate output logits
        return self.output_layer(x)

# 3. Training function
def train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# 4. Example usage
def main():
    # Sample data
    input_lines = [
        "def func(x):",
        "for i in range(n):"
    ]
    target_lines = [
        "return x * x",
        "sum += i"
    ]
    
    # You'll need to install and import transformers library
    from transformers import RobertaTokenizer
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Create dataset and dataloader
    dataset = CodeCompletionDataset(input_lines, target_lines, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CodeCompletionModel(vocab_size=tokenizer.vocab_size).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    train_model(model, dataloader, optimizer, criterion, device)

if __name__ == "__main__":
    main()

# use: -----
def generate_completion(model, tokenizer, input_text: str, device):
    model.eval()
    with torch.no_grad():
        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        # Generate output
        outputs = model(input_ids)

        # Get the most likely tokens
        predicted_ids = torch.argmax(outputs[0], dim=-1)

        # Decode the prediction
        predicted_text = tokenizer.decode(predicted_ids)

        return predicted_text

# Example usage
input_text = "def func(x):"
#completion = generate_completion(model, tokenizer, input_text, device)
#print(f"Input: {input_text}")
#print(f"Completion: {completion}")

