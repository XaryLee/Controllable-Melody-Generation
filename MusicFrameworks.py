import copy
import numpy as np
from tqdm import tqdm

def read_melody(filename):
    # key_file = filename + 'analyzed_key.txt'
    filename += 'melody.txt'
    # with open(key_file, 'r') as f:
    #     key_sig = f.readline()
    #     key_sig = key_sig.split(':')
        
    notes = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            pitch = eval(line[0])
            duration = eval(line[1])
            notes.append([pitch, duration])
    # notes = zero_process(notes)
    return notes

def read_chord(filename):
    '''
    return list
    '''
    chords = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            beg = line.find('[')
            end = line.find(']')
            chord = line[beg: end+1]
            chord_idx = eval(chord)
            chord = np.zeros(13)
            chord[chord_idx] = 1
            duration = eval(line.split()[-1])
            duration *= 4 # 数据集中和弦的 1 duration是 1 beat 也就是4个16分音符
            chord[-1] = duration
            # 由于数据集中chords中可能包含3个音可能包含4个音，所以统一处理成12维one-hot+1维duration
            chord = list(chord)
            chords.append(chord)
    return chords

def zero_process(section): # 处理一个section内部的休止符
    '''
    发现数据集内部有很多0，但实际音乐中应该处理为前一个音的延伸而不是休止
    后续考虑到有间奏、弱起等，休止并不是完全无用的，还是做了保留
    '''
    section_new = []
    for note in section:
        pitch, duration = note
        if pitch == 0 and len(section_new) > 0:
            section_new[-1][1] += duration
        else:
            section_new.append([pitch, duration])
        
    return section_new

def two_beat_split(section):
    # 16分音符为一个单位，duration=4是1拍，duration=8是2拍
    half_measures = [] # 记录2-beat组
    cur_notes = []
    i = 0
    t = 0
    while i < len(section):
        note = section[i]
        pitch, duration = note
        if t < 8:
            if t + duration <= 8:
                t += duration
                cur_notes.append(note)
                i += 1
                # print(i, end=' ')
            else:
                cur_notes.append([pitch, 8-t])
                section[i][-1] -= (8-t)
                t = 8
            # 横跨多个2-beat的处理
        if t >= 8:
            half_measures.append(cur_notes)
            t = 0
            cur_notes = []
    # 剔除intro
    half_measures_new = []
    i = 0
    while half_measures[i] == [[0, 8]]:
        i += 1
    half_measures_new.extend(half_measures[i:])
        
    return half_measures_new, i

def two_beat_split_chord(section):
    # 16分音符为一个单位，duration=4是1拍，duration=8是2拍
    half_measures = [] # 记录2-beat组
    cur_notes = []
    i = 0
    t = 0
    while i < len(section):
        note = section[i]
        pitch = note[:-1]
        duration = note[-1]
        if t < 8:
            if t + duration <= 8:
                t += duration
                cur_notes.append(note)
                i += 1
                # print(i, end=' ')
            else:
                cur_notes.append(pitch + [8-t])
                section[i][-1] -= (8-t)
                t = 8
            # 横跨多个2-beat的处理
        if t >= 8:
            half_measures.append(cur_notes)
            t = 0
            cur_notes = []
    return half_measures

def get_basic_melody(half_measures):
    basic_melody = []
    for half_measure in half_measures:
        notes_dict = {0: 0.5}
        for note in half_measure:
            pitch, duration = note
            if pitch: # 0不算入basic_melody
                if pitch not in notes_dict.keys(): # 0不算入basic_melody
                    notes_dict[pitch] = duration
                else:
                    notes_dict[pitch] += duration
        basic_note = max(notes_dict, key=notes_dict.get)
        basic_melody.append(basic_note)
    return basic_melody

# def get_basic_rhythm(half_measures, thresh=0.85):
#     complexity_list = []
#     pattern_tags = []
#     rhythm_pattern = []
#     # 一般来说流行音乐的节奏变化不是很大，不会出现AB相似，BC相似，但AC不相似的情况
#     for half_measure in half_measures:
#         is_exist = False
#         complexity = len(half_measure) / 16
#         complexity_list.append(complexity)
#         for tag, pattern in enumerate(rhythm_pattern):
#             rscore = (rhythm_similarity(pattern, half_measure) + rhythm_similarity(half_measure, pattern)) / 2
#             pattern1 = copy.deepcopy(pattern)
#             pattern1.insert(0, [0, 1])
#             half_measure1 = copy.deepcopy(half_measure)
#             half_measure1.insert(0, [0, 1])
#             # 根据论文描述。数据集中可能存在16分音符的偏移，所以加上偏移算sim取最大
#             rscore1 = rhythm_similarity(pattern1, half_measure)
#             rscore2 = rhythm_similarity(pattern, half_measure1)
#             rscore = max(rscore, rscore1, rscore2)
#             # print(rscore)
#             if rscore >= thresh:
#                 is_exist = True
#                 pattern_tags.append(tag)
#                 break
#         if not is_exist:
#             # print('11111')
#             pattern_tags.append(len(rhythm_pattern))
#             rhythm_pattern.append(half_measure)
#     return pattern_tags, complexity_list

def get_basic_rhythm(half_measures, thresh=0.85):
    complexity_list = []
    pattern_tags = []
    rhythm_pattern = []
    rhythm_pattern_idx = [] # 与rhythm pattern等长，记录最先出现这个pattern的位置
    pattern_position = []
    # 一般来说流行音乐的节奏变化不是很大，不会出现AB相似，BC相似，但AC不相似的情况
    for i, half_measure in enumerate(half_measures):
        is_exist = False
        complexity = len(half_measure) / 16
        complexity_list.append(complexity)
        for tag, pattern in enumerate(rhythm_pattern):
            rscore = (rhythm_similarity(pattern, half_measure) + rhythm_similarity(half_measure, pattern)) / 2
            pattern1 = copy.deepcopy(pattern)
            pattern1.insert(0, [0, 1])
            half_measure1 = copy.deepcopy(half_measure)
            half_measure1.insert(0, [0, 1])
            # 根据论文描述。数据集中可能存在16分音符的偏移，所以加上偏移算sim取最大
            rscore1 = rhythm_similarity(pattern1, half_measure)
            rscore2 = rhythm_similarity(pattern, half_measure1)
            rscore = max(rscore, rscore1, rscore2)
            # print(rscore)
            if rscore >= thresh:
                is_exist = True
                pattern_tags.append(tag)
                pattern_position.append(rhythm_pattern_idx[tag])
                break
        if not is_exist:
            # print('11111')
            pattern_tags.append(len(rhythm_pattern))
            rhythm_pattern.append(half_measure)
            rhythm_pattern_idx.append(i)
            pattern_position.append(i)
            
    return pattern_tags, complexity_list, pattern_position

def rhythm_similarity(x, y):
    lx = len(x)
    ly = len(y)
    t1 = 0
    t2 = 0
    i = 0
    j = 0
    t = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # for (int t = 0; i < lx && j < ly; ++t) { # 以最小单元为时间轴移动，不断添加音符进去，并计算两个序列是否执行相同动作
    while i < lx and j < ly:
        xpitch, xduration = x[i]
        ypitch, yduration = y[j]
        if t1 + xduration <= t:
            t1 += xduration
            i += 1
            b = 0
        else:
            b = -1
        if xpitch == 0:
            b = -1
        if t2 + yduration <= t:
            t2 += yduration
            j += 1
            a = 0
        else:
            a = -1
        if ypitch == 0:
            a = -1
        
        if a == b:
            if a == 0:
                tp += 1
            else:
                tn += 1
        elif a == 0:
            fp += 1
        else:
            fn += 1
        t += 1
    acc = 1
    if tp + tn + fp + fn > 0:
        acc = (tp + tn) / (tp + tn + fp + fn)
    return acc

# def read_all_melody(filedir):
#     for i in tqdm(range(1, 910)):
#         songid = str(i).zfill(3)
#         filename = filedir + '/' + songid + '/' + 'melody.txt'
#         notes = read_melody(filename)
#         # print(notes)
#         half_measures = two_beat_split(notes)
#         # print(half_measures)
#         basic_melody = get_basic_melody(half_measures)
#         # print(basic_melody)
#         # print(rhythm_similarity(half_measures[7], half_measures[8]))
#         # print(rhythm_similarity(half_measures[8], half_measures[7]))
#         result1, result2 = get_basic_rhythm(half_measures)
#         # print(list(zip(result1, result2)))
#         # exit()

if __name__ == '__main__':
    filedir = 'POP909'
    
    # read all melody
    for i in tqdm(range(1, 910)):
        songid = str(i).zfill(3)
        filename = filedir + '/' + songid + '/'
        notes = read_melody(filename)
        print(notes)
        half_measures, offset = two_beat_split(notes)
        print(half_measures)
        basic_melody = get_basic_melody(half_measures)
        print(basic_melody)
        # print(rhythm_similarity(half_measures[7], half_measures[8]))
        # print(rhythm_similarity(half_measures[8], half_measures[7]))
        # result1, result2 = get_basic_rhythm(half_measures)
        # print(list(zip(result1, result2)))
        exit()
        
    # read all chord
    for i in tqdm(range(1, 910)):
        songid = str(i).zfill(3)
        filename = filedir + '/' + songid + '/' + 'finalized_chord.txt'
        chords = read_chord(filename)
        chords_half_measures = two_beat_split_chord(chords)
        if i == 1:
            print(chords)
            print(chords_half_measures)
    print('done')