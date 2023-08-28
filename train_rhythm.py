import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os

from model import TransformerLSTM
from dataloader import RhythmDataloader

# train函数直接从melody那里copy过来的，只改了参数，一些变量名没有改

os.system('rm rhythm_logs/*')
writer = SummaryWriter('rhythm_logs')

## parameters
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_model = {
    'input_size': 4, # brf + pos, 2 + 2
    'feedforward_size': 128,
    'nhead': 8,
    'LSTM_hidden_size': 64,
    'output_size': 8,
    'dropout': 0.1
}
lr = 1e-5

params_loader = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 0
}

input_size = config_model['input_size']
feedforward_size = config_model['feedforward_size']
nhead = config_model['nhead']
dropout = config_model['dropout']
LSTM_hidden_size = config_model['LSTM_hidden_size']
output_size = config_model['output_size']

batch_size = params_loader['batch_size']

loader = RhythmDataloader(params_loader)
model = TransformerLSTM(config_model)
loss_fn = nn.CrossEntropyLoss()

model = model.to(device)
loss_fn = loss_fn.to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 使用Adam默认参数
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

print('start training')
for epoch in range(epochs):
    print(f'-----epoch {epoch+1}-----')
    model.train()
    for i, data in enumerate((loader)):
        # basic_melody: tensor, chords: list
        basic_melody, xseq = data
        # basic_melody = basic_melody.squeeze()
        # xseq = xseq.squeeze()
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
        loss = loss_fn(output, basic_melody[0:seqlen])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('train_loss', loss, epoch*len(loader)+i)
        
        if i % 20 == 0:
            print(loss)
            
    scheduler.step()

writer.close()
save_name = 'rhythm.pth'
torch.save(model, 'save_model/' + save_name)
print('training completed')
            