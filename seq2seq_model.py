import torch.nn as nn
import torch 

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        # Encoder
        embedded = self.embedding(input_seq)
        _, (hidden, cell) = self.encoder(embedded)

        # Decoder
        decoder_input = input_seq[:, 0].unsqueeze(1)  # Start token
        outputs = []
        max_len = target_seq.size(1) if target_seq is not None else 50  # Use max_len for inference
        for t in range(max_len):
            decoder_embedded = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = output.argmax(2)
        outputs = torch.cat(outputs, dim=1)
        return outputs

