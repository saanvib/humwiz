import MIDI_Pitch_Extraction as midi
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
import pitchDetection as pitch
import os
import pickle

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


def addtodata(filepath,label, array_list):
    array_list.append(labels(finaloutput(filepath, 30), label))
    return array_list
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

def upload(curr_path, array_l):
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']
    for folder in os.listdir(curr_path):
        fol = os.path.join(curr_path, folder)
        if not os.path.isdir(fol):
            continue
        for file in os.listdir(fol):
            f = os.path.join(fol, file)
            array_l = addtodata(str(f),NOTES.index(str(folder)), array_l)
    return array_l
        
def trainModel():
    array_list = []
    array_list = upload("MIDIS/", array_list)


    dataset = np.concatenate([i[0] for i in array_list], axis = 0)
    dlabels = numpy.concatenate([i[1] for i in array_list], axis = 0)

    column_values = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']

    X = dataset
    Y = dlabels
    table = pd.DataFrame(data=dataset, index=dlabels, columns=column_values, dtype=None, copy=None)
    # print(table)
    #print('X', X.shape, 'Y', Y.shape)
    clf = RandomForestClassifier()
    clf = clf.fit(X, Y)  
    with open("classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("dumped pickle")

    
def main(file_n, clf):
    print("calling main")
    # plt.figure(figsize=(30,10), facecolor ='k')
    # a = tree.plot_tree(clf,

    #                 feature_names = column_values,

    #                 class_names = [str(i) for i in dlabels],

    #                 rounded = True,

    #                 filled = True,

    #                 fontsize=14)
    # # plt.show()



    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']
    CORRESPONDING_MINOR = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    midi_notes_arr = pitch.extractPitch(file_n)
    print("about to load model")
    clf = pickle.load(open("classifier.pkl", "rb"))
    pred = clf.predict(finaloutput(midi_notes_arr, 10000))
    # print(pred)
    print(NOTES[pred[0]])
    end_list = []
    end_list.append(NOTES[pred[0]])
    end_list.append(CORRESPONDING_MINOR[pred[0]])
    return end_list

def testOnFiles():
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A', 'A#','B']
    folder_path = "TestHumFiles/"
    output_results = open("results.txt", "w")
    correct_count = 0
    tot_count = 0
    model = trainModel()
    # predictions = np.zeros(72)
    # actual = np.zeros(72)
    max_error_key = 0
    max_error_count = 0
    for folder in os.listdir(folder_path):
        fol = os.path.join(folder_path, folder)
        error_count = 0
        if not os.path.isdir(fol):
            continue
        for file in os.listdir(fol):
            root, extension = os.path.splitext(file)
            if (extension == ".m4a"):
                f = os.path.join(fol, file)
                res = main(f, model)
                s = "expected " + str(folder) + ", got " + str(res[0])
                if str(folder) == str(res[0]):
                    correct_count += 1
                elif abs(NOTES.index(str(folder))-NOTES.index(str(res[0]))) == 1 or abs(NOTES.index(str(res[0]))-NOTES.index(str(folder))) == 1:
                    correct_count += 0.5
                elif abs(NOTES.index(str(folder))-NOTES.index(str(res[0]))) == 2 or abs(NOTES.index(str(res[0]))-NOTES.index(str(folder))) == 2:
                    correct_count += 0.2
                else:
                    error_count += 1
                tot_count += 1
                #make integers
                # predictions = predictions.insert((tot_count-1), NOTES.index(str(res[0])))
                # actual = actual.insert((tot_count-1), NOTES.index(str(folder)))
                print(s)
                output_results.write(s + "\n")
        if (error_count > max_error_count):
            max_error_key = NOTES.index(str(folder))
            max_error_count = error_count
    print("most mistakes in " + NOTES[max_error_key])
    print(str(correct_count) + " correct out of " + str(tot_count))
    print(str(((correct_count)/tot_count)))
    # print(confusion_matrix(actual, predictions))
    output_results.write(str(correct_count) + " correct out of " + str(tot_count) + "\n")
    output_results.close()        
        
# testOnFiles()

def finalFunc(f):
    print("calling final func")
    # model = trainModel()
    model = pickle.load(open("classifier.pkl", "rb"))
    print("trained model")
    return main(f, model)


# clf = pickle.load(open("classifier.pkl", "rb"))
# main("TestHumFiles/A/HumA.m4a", clf)