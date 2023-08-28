from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from MusicFrameworks import *

class POP909BasicMelodyDataset(Dataset):
    def __init__(self, filedir='POP909', alignlen=64):
        self.basic_melody_list = []
        self.chords_list = []
        self.alignlen = alignlen
        print('开始构建 POP909 basic melody dataset ···')
        for i in tqdm(range(1, 910)):
            songid = str(i).zfill(3)
            filename = filedir + '/' + songid + '/'
            notes = read_melody(filename)
            half_measures, offset = two_beat_split(notes)
            basic_melody = get_basic_melody(half_measures)
            
            filename = filedir + '/' + songid + '/' + 'finalized_chord.txt'
            chords = read_chord(filename)
            chords_half_measures = two_beat_split_chord(chords)
            chords_half_measures = chords_half_measures[offset:]
            
            seqlen = min(len(basic_melody), len(chords))
            
            if self.alignlen > 0:
                if seqlen >= self.alignlen:
                    self.chords_list.append(chords_half_measures[:self.alignlen])
                    self.basic_melody_list.append(basic_melody[:self.alignlen])
        print('done')
    
    def __getitem__(self, index):
        '''
        返回一条basic_melody和它对应的chord_seq
        '''
        basic_melody = self.basic_melody_list[index]
        # basic_melody = torch.FloatTensor(basic_melody)
        
        chords = self.chords_list[index]
        # chords = torch.FloatTensor(chords)
        # 每个half_measure中的chord数不一致，只能以列表形式返回
        
        # print(len(basic_melody), len(chords))
        seqlen = min(len(basic_melody), len(chords))
        
        xseq = []
        notes_label = []
        
        note_0_idx = int(basic_melody[0])
        note_0 = torch.zeros(128)
        note_0[note_0_idx] = 1
        
        notes_label.append(note_0)
        
        # note转ont-hot编码
        chord_0 = chords[0]
        # chord_0是一个half_measure
        # (num_layers, output_size), (num_layers, hidden_size)
        cur_chord = chord_0[0]
        pre_chord = chord_0[0]
        if len(chord_0) == 1:
            if len(chords) > 1:
                nxt_chord = chords[1][0] # 下一个half_measure的第一个chord
            else:
                nxt_chord = chord_0[0]
        else:
            nxt_chord = chord_0[1]
        x = pre_chord + cur_chord + nxt_chord + [0]
        xseq.append(x)
        # x = torch.FloatTensor(x)
        
        # x = x.to(device)
        # note_0 = note_0.to(device)
        # pre_output = torch.zeros(output_size).to(device)
        
        # output_0, (h_n, c_n) = model(x, pre_output, (h_0, c_0))
        # pre_output = output_0
        # loss = loss_fn(output_0, note_0)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        for idx in range(1, seqlen):
            # idx is position
            note_idx = int(basic_melody[idx])
            note = torch.zeros(128)
            note[note_idx] = 1
            
            notes_label.append(note)
            chord = chords[idx]
            
            cur_chord = chord[0]
            pre_chord = chords[idx-1][-1]
            if len(chord) == 1:
                if idx < len(chords)-1:
                    nxt_chord = chords[idx+1][0] # 下一个half_measure的第一个chord
                else:
                    nxt_chord = chord[0]
            else:
                nxt_chord = chord[1]
            x = pre_chord + cur_chord + nxt_chord + [idx]
            xseq.append(x)
            
        xseq = np.array(xseq)
        xseq = torch.FloatTensor(xseq).squeeze()
        basic_melody = np.array(basic_melody)
        basic_melody = torch.FloatTensor(basic_melody).squeeze()
        
        notes_label = torch.stack(notes_label)
        
        return notes_label, xseq
    
    def __len__(self):
        return len(self.basic_melody_list)
    
class POP909BasicMelodyDataloader(DataLoader):
    def __init__(self, params, filedir='POP909'):
        dataset = POP909BasicMelodyDataset(filedir)
        
        batch_size = params['batch_size']
        shuffle = params['shuffle']
        num_workers = params['num_workers']
        
        if batch_size == None:
            self.init_kwargs = {
                'dataset': dataset,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        else:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        super().__init__(**self.init_kwargs)


class RhythmDataset(Dataset):
    def __init__(self, filedir='POP909', alignlen=64):
        self.basic_melody_list = []
        self.chords_list = []
        self.alignlen = alignlen
        brf_all_data = []
        pattern_vec_all_data = [] # 用8维向量表示每个pattern 1表示音符onset
        print('开始构建 POP909 rhythm dataset ···')
        for i in tqdm(range(1, 910)):
            songid = str(i).zfill(3)
            filename = filedir + '/' + songid + '/'
            notes = read_melody(filename)
            half_measures, offset = two_beat_split(notes)
            pattern_tags, complexity_list, pattern_position = get_basic_rhythm(half_measures)
            brf_list = list(zip(pattern_position, complexity_list))
            
            pattern_vec_list = []
            for half_measure in half_measures:
                pattern_vec = [1] + [0] * 7
                total_dura = 0
                for note in half_measure:
                    duration = note[1]
                    total_dura += duration
                    idx = min(total_dura, 7)
                    pattern_vec[idx] = 1
                pattern_vec_list.append(pattern_vec)
            
            seqlen = len(complexity_list)
            
            if self.alignlen > 0:
                if seqlen >= self.alignlen:
                    brf_all_data.append(brf_list[:self.alignlen])
                    pattern_vec_all_data.append(pattern_vec_list[:self.alignlen])
                    
            self.brf_all_data = brf_all_data
            self.pattern_vec_all_data = pattern_vec_all_data
            
        print('done')
    
    def __getitem__(self, index):
        brfs = self.brf_all_data[index]
        pattern_vecs = self.pattern_vec_all_data[index]
        
        seqlen = len(brfs)
        positions = np.arange(seqlen)
        is_on_barline = positions % 2 # 0是在小节线上
        positions = torch.FloatTensor(positions)
        positions = positions.unsqueeze(1)
        is_on_barline = torch.FloatTensor(is_on_barline)
        is_on_barline = is_on_barline.unsqueeze(1)
        brfs = np.array(brfs)
        brfs = torch.FloatTensor(brfs)
        features = torch.concat((brfs, positions, is_on_barline), dim=1)
        
        label = np.array(pattern_vecs)
        label = torch.FloatTensor(label)
        
        return label, features
    
    def __len__(self):
        return len(self.brf_all_data)
    
class RhythmDataloader(DataLoader):
    def __init__(self, params, filedir='POP909'):
        dataset = RhythmDataset(filedir)
        
        batch_size = params['batch_size']
        shuffle = params['shuffle']
        num_workers = params['num_workers']
        
        if batch_size == None:
            self.init_kwargs = {
                'dataset': dataset,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        else:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        super().__init__(**self.init_kwargs)
        
class MelodyDataset(Dataset):
    
    def __init__(self, filedir='POP909', alignlen=64, notenum=32, seglen=8):
        '''
        alignlen: 截断长度
        notenum: 音符对齐长度（每个片段包含的音符数量）
        seglen: 片段长度
        一个序列包含多个片段 以片段为训练基本单位
        '''
        self.notenum = notenum
        self.seglen = seglen
        
        self.pitches_all_data = []
        self.durations_all_data = []
        self.basic_pitches_all_data = []
        self.chords_all_data = []
        self.bar_positions_all_data = []
        self.alignlen = alignlen
        print('开始构建 POP909 melody dataset ···')
        for i in tqdm(range(1, 910)):
            songid = str(i).zfill(3)
            filename = filedir + '/' + songid + '/'
            notes = read_melody(filename)
            pitches = []
            durations = []
            basic_pitches = []
            basic_chords = []
            bar_positions = []
            # for note in notes:
            #     pitches.append(note[0])
            #     durations.append(note[1])
            
            half_measures, offset = two_beat_split(notes)
            basic_melody = get_basic_melody(half_measures)
            
            filename = filedir + '/' + songid + '/' + 'finalized_chord.txt'
            chords = read_chord(filename)
            chords_half_measures = two_beat_split_chord(chords)
            chords_half_measures = chords_half_measures[offset:]
            
            # 以2-beat为单位对齐note和basic pitch
            seqlen = min(len(half_measures), len(chords_half_measures))
            if self.alignlen > 0:
                if seqlen >= self.alignlen:
                    seqlen = self.alignlen
                else:
                    continue
            prej = 0
            j = 0
            for i in range(seqlen):
                if i % 2 == 0:
                    prej = 0
                basic_pitch = basic_melody[i]
                half_measure = half_measures[i]
                basic_chord = chords_half_measures[i][0][:-1]
                for j, note in enumerate(half_measure):
                    pitches.append(note[0])
                    durations.append(note[1])
                    basic_pitches.append(basic_pitch)
                    basic_chords.append(basic_chord)
                    bar_positions.append(prej+j)
                prej = j
            
            # pattern_vec_list = []
            # for half_measure in half_measures:
            #     pattern_vec = [1] + [0] * 7
            #     total_dura = 0
            #     for note in half_measure:
            #         duration = note[1]
            #         total_dura += duration
            #         idx = min(total_dura, 7)
            #         pattern_vec[idx] = 1
            #     pattern_vec_list.append(pattern_vec)

                # 每达到一个片段就保存
                if (i+1) % self.seglen == 0:
                    self.pitches_all_data.append(pitches)
                    self.durations_all_data.append(durations)
                    self.basic_pitches_all_data.append(basic_pitches)
                    self.chords_all_data.append(basic_chords)
                    self.bar_positions_all_data.append(bar_positions)
            
        print('done')
    
    def __getitem__(self, index):
        pitches = self.pitches_all_data[index]
        durations = self.durations_all_data[index]
        basic_pitches = self.basic_pitches_all_data[index]
        chords = self.chords_all_data[index]
        bar_positions = self.bar_positions_all_data[index]
        
        seqlen = len(pitches)
        positions = np.arange(seqlen)
        positions = torch.FloatTensor(positions)
        positions = positions.unsqueeze(1)
        durations = np.array(durations)
        durations = torch.FloatTensor(durations).unsqueeze(1)
        basic_pitches = np.array(basic_pitches)
        basic_pitches = torch.FloatTensor(basic_pitches).unsqueeze(1)
        chords = np.array(chords)
        chords = torch.FloatTensor(chords)
        bar_positions = np.array(bar_positions)
        bar_positions = torch.FloatTensor(bar_positions).unsqueeze(1)
        features = torch.concat((durations, basic_pitches, chords, bar_positions, positions), dim=1)
        
        pitches_one_hot = []
        for pitch in pitches:
            one_hot = torch.zeros(128).float()
            one_hot[pitch] = 1
            pitches_one_hot.append(one_hot)
        label = torch.stack(pitches_one_hot)
        
        # print(label.shape, features.shape)
        
        if len(label) >= self.notenum:
            label = label[:self.notenum]
            features = features[:self.notenum]
            pad_offset = self.notenum
        else:
            pad_offset = len(label)
            pad = torch.zeros((self.notenum-len(label), 128))
            label = torch.concat((label, pad), dim=0)
            pad = torch.zeros((self.notenum-len(features), 16))
            features = torch.concat((features, pad), dim=0)
        
        return label, features, pad_offset
    
    def __len__(self):
        return len(self.pitches_all_data)

class MelodyDataloader(DataLoader):
    def __init__(self, params, filedir='POP909'):
        dataset = MelodyDataset(filedir)
        
        batch_size = params['batch_size']
        shuffle = params['shuffle']
        num_workers = params['num_workers']
        
        if batch_size == None:
            self.init_kwargs = {
                'dataset': dataset,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        else:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'drop_last': True
            }
        super().__init__(**self.init_kwargs)
        
if __name__ == '__main__':
    # parameters
    params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 0
    }
    # dataset = POP909BasicMelodyDataset()
    # loader = POP909BasicMelodyDataloader(params)
    # dataset = MelodyDataset()
    # loader = RhythmDataloader(params)
    loader = MelodyDataloader(params)
    # print(dataset[0])
    # len_dict = {}
    # for data in loader:
    #     basic_melody, xseq = data
    #     seqlen = min(len(basic_melody[0]), len(xseq[0]))
    #     seqlen = int(seqlen / 10)
    #     if seqlen not in len_dict.keys():
    #         len_dict[seqlen] = 1
    #     else:
    #         len_dict[seqlen] += 1
    # print(len_dict)
    # for data in loader:
    #     basic_melody, xseq = data
    #     print(basic_melody.shape, xseq.shape)
    #     print(basic_melody)
    #     print(xseq)
    #     break
    len_dict = {}
    for i, data in enumerate(loader):
        # print(data)
        label, features = data
        print(label.shape, features.shape)
        print(data)
        print(len(loader))
        if i >= 0:
            break