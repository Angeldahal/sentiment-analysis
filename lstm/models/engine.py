import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_size,
            n_layers,
            bidirectional,
            dropout_rate,
            pad_index
            ):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch_size, seq_len]
        # length = [batch_size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch_size, seq_len, embedding_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n_layers * n_directions, batch_size, hidden_dim]
        # cell = [n_layers * n_directions, batch_size, hidden_dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch_size, seq_len, hidden_dim*n_directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2 ]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch_size, output_dim]
        return prediction