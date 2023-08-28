import torch
from torch import nn

# parameters
config = {
    'input_size': 40, # chord + pos, 13 * 3 + 1
    'feedforward_size': 128,
    'nhead': 8,
    'LSTM_hidden_size': 64,
    'output_size': 128,
    'dropout': 0.2
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerLSTM(nn.Module):
    
    def __init__(self, config=config):
        
        super().__init__()
        
        input_size = config['input_size']
        feedforward_size = config['feedforward_size']
        nhead = config['nhead']
        dropout = config['dropout']
        LSTM_hidden_size = config['LSTM_hidden_size']
        output_size = config['output_size']
        
        self.feedforward_size = feedforward_size
        
        self.feedforward = nn.Linear(input_size, feedforward_size)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=feedforward_size,
                                                               nhead=nhead,
                                                               dim_feedforward=feedforward_size,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                         num_layers=2)
        self.lstm = nn.LSTMCell(input_size=2*feedforward_size,
                                hidden_size=LSTM_hidden_size)
        self.conv_layers = nn.Sequential(nn.Conv1d(LSTM_hidden_size, output_size, 1),
                                         nn.ReLU(),
                                         nn.Conv1d(output_size, output_size, 1))
        self.softmax = nn.Softmax(dim=1)
        # lstm output size是64，auto-regression length=1，size=[1, 64]
        self.embedding = nn.Linear(output_size, feedforward_size)
        
    def forward(self, x, idx, pre_output, memory):
        '''
        auto-regression
        input: sequence x & previous output note
        idx: select idx_th code after encoder to decode
        output: single note y
        memory: LSTM hidden state, ((num_layers, output_size), (num_layers, hidden_size))
        x: (seqlen, batch_size, dmodel)
        '''
        ff = self.feedforward(x[0])
        ff = ff.unsqueeze(0)
        for xx in x[1:]:
            xx = self.feedforward(xx)
            xx = xx.unsqueeze(0)
            ff = torch.concat((ff, xx), 0)
        x = ff
        x = self.transformer_encoder(x)
        e = self.embedding(pre_output)
        x = x[idx]
        x = torch.concat((x, e), dim=1)
        x, cx = self.lstm(x, memory)
        memory = (x, cx)
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = x.squeeze(dim=1)
        output = self.softmax(x)
        
        return output, memory