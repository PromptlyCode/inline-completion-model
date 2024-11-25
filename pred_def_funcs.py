import torch
import torch.nn as nn
import os
import re
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# Custom dataset for code sequences
class CodeDataset(Dataset):
    def __init__(self, code_files_path, max_length=512):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        
        self.samples = []
        self.function_names = []
        
        # Process all Python files in the directory
        for root, _, files in os.walk(code_files_path):
            for file in files:
                if file.endswith('.py'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract function definitions
                        functions = re.finditer(r'def\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*:', content)
                        for match in functions:
                            func_name = match.group(1)
                            self.function_names.append(func_name)
                            # Get the function body
                            start = match.start()
                            next_def = content.find('def', match.end())
                            end = next_def if next_def != -1 else len(content)
                            func_code = content[start:end].strip()
                            self.samples.append(func_code)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code = self.samples[idx]
        encoded = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'function_name': self.function_names[idx]
        }

# Transformer model for code completion
class CodeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1024, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, attention_mask=None):
        # Create position indices
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        # Combine token embeddings and positional encodings
        x = self.embedding(x) + self.pos_encoder(positions)
        
        # Create attention mask for padding tokens
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Project to vocabulary size
        output = self.fc(x)
        return output

# Training function
def train_model(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift sequences for next token prediction
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Reshape outputs and targets for loss calculation
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

# Function to generate code completion
def generate_completion(model, tokenizer, prompt, max_length=100, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset and dataloader
    dataset = CodeDataset('path_to_your_python_files')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model and training components
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = CodeTransformer(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    train_model(model, train_loader, optimizer, criterion, device)
    
    # Example completion
    prompt = "def"
    completion = generate_completion(model, tokenizer, prompt, device=device)
    print(f"Generated completion:\n{completion}")

if __name__ == "__main__":
    main()

