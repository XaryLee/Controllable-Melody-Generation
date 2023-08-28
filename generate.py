import torch
from dataloader import MelodyDataset

config_model = {
    'input_size': 16, 
    'feedforward_size': 128,
    'nhead': 8,
    'LSTM_hidden_size': 64,
    'output_size': 128,
    'dropout': 0.2
}

input_size = config_model['input_size']
feedforward_size = config_model['feedforward_size']
nhead = config_model['nhead']
dropout = config_model['dropout']
LSTM_hidden_size = config_model['LSTM_hidden_size']
output_size = config_model['output_size']

batch_size = 1

save_name = 'melody.pth'
model = torch.load('save_model/' + save_name)
dataset = MelodyDataset()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

idx = 1
data = dataset[idx]

basic_melody, xseq = data
basic_melody = basic_melody.unsqueeze(0)
xseq = xseq.unsqueeze(0)
xseq = xseq.to(device)
basic_melody = basic_melody.to(device)

# print(len(basic_melody), len(chords))
seqlen = min(len(basic_melody[0]), len(xseq[0]))
basic_melody = basic_melody.transpose(0, 1)
xseq = xseq.transpose(0, 1)
output_seq = []

h_0 = torch.zeros((batch_size, LSTM_hidden_size)).to(device)
c_0 = torch.zeros((batch_size, LSTM_hidden_size)).to(device)

pre_output = torch.zeros((batch_size, output_size)).to(device)

output_0, (h_n, c_n) = model(xseq, 0, pre_output, (h_0, c_0))
pre_output = output_0
output_seq.append(output_0)

for idx in range(1, seqlen):
    
    output, (h_n, c_n) = model(xseq, idx, pre_output, (h_n, c_n))
    pre_output = output
    output_seq.append(output)

output = torch.stack(output_seq, 0).to(device)
output = output.squeeze()
output = torch.argmax(output, dim=1)
print(output)