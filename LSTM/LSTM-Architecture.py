class LSTMModel(nn.Module):
    def __init__(self, hid_dim, emb_dim, vocab_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size+1
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hid_dim,
                            batch_first=True, num_layers=self.num_layers)
        self.drop = nn.Dropout(0.3)
        # from here we will randomly sample a word
        self.linear = nn.Linear(self.hid_dim, vocab_size)

    def forward(self, x, prev_hid):
        x = self.embedding(x)
        x, hid = self.lstm(x, prev_hid)
        x = self.drop(x)
        x = self.linear(x)
        return x, hid

    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hid_dim), torch.zeros(self.num_layers, batch_size, self.hid_dim))
