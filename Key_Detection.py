import MIDI_Pitch_Extraction as midi
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree 
import pitchDetection as pitch
import os

def splitsongs(input, size):
    if not isinstance(input, list):
        input = midi.songProcessing(input)
    steps = []
    val = 0
    sz = size
    b = len(input)
    if sz >= b:
        steps.append(input)
        return steps
    for i in range(0,b,5):
        chunk = input[i: sz + i]
        if len(chunk) == sz:
            steps.append(chunk)            
    return steps

def makehist(inputfile):
    newsong = []
    song = (inputfile)#midi.songProcessing(inputfile)
    #print(len(song))
    for i in range(len(song)):
        newsong.append(song[i] % 12)
    #print(newsong)
    x = numpy.array(newsong)
    bin_edges=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
    histo = numpy.histogram(x, bins=bin_edges, range=None, density=None, weights=None)

    _ = plt.hist(x, bins=bin_edges)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
   # plt.show()
    #print("histo", histo)
    return histo

def sort(histogram):
    counts = histogram[0]
    bins = histogram[1]
    # print(counts)
    # print(np.sum(counts))
    counts = counts / np.sum(counts)
    #print(counts)
    return counts

def finaloutput(inputt, sizee):
    probs = []
    x = splitsongs(inputt,sizee)
    for i in x:
        h = makehist(i)
        b = sort(h)
        probs.append(b)
    return np.array(probs)

def labels(fin, lbl):
    labs = []
    
    clip_num = fin.shape[0]
    for i in range(int(fin.shape[0])):
        labs.append(lbl)
    #lb = np.ones((clip_num,1)) * lbl 
    #print(fin)
    np.ndarray.transpose(fin)
    labs = np.array(labs)
    return fin, labs

array_list = []
def addtodata(filepath,label):
    array_list.append(labels(finaloutput(filepath, 30), label))
"""       LABELS
0 = C MAJOR/ A MINOR
1 = C# MAJOR/ A# MINOR
2 = D MAJOR/ B MINOR
3 = D# MAJOR/ C MINOR
4 = E MAJOR/ C# MINOR
5 = F MAJOR/ D MINOR
6 = F# MAJOR/ D# MINOR
7 = G MAJOR/ E MINOR
8 = G# MAJOR/ F MINOR
9 = A MAJOR/ F# MINOR
10 = A# MAJOR/ G MINOR
11 = B MAJOR/ G# MINOR """

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']
NUMS = [i for i in range(12)]
def upload(curr_path):
    for folder in os.listdir(curr_path):
        fol = os.path.join(curr_path, folder)
        for file in os.listdir(fol):
            f = os.path.join(fol, file)
            addtodata(str(f),NOTES.index(str(folder)))
        
upload("MIDIS/")

dataset = np.concatenate([i[0] for i in array_list], axis = 0)
dlabels = numpy.concatenate([i[1] for i in array_list], axis = 0)

column_values = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']

X = dataset
Y = dlabels
table = pd.DataFrame(data=dataset, index=dlabels, columns=column_values, dtype=None, copy=None)
print(table)
#print('X', X.shape, 'Y', Y.shape)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)  
plt.figure(figsize=(30,10), facecolor ='k')
a = tree.plot_tree(clf,

                   feature_names = column_values,

                   class_names = [str(i) for i in dlabels],

                   rounded = True,

                   filled = True,

                   fontsize=14)
plt.show()

midi_notes_arr = pitch.extractPitch("nonCodeFiles/New Recording 47.m4a")

pred = clf.predict(finaloutput(midi_notes_arr, 10000))
print(pred)
print(NOTES[pred[0]])