import numpy as np
import essentia
import essentia.standard as ess
import IPython.display as ipd
import os
import math
from pydub import AudioSegment

file_path = os.path.join("Creepin Piano.m4a")

EXPECTED_SAMPLE_RATE = 16000

def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  audio.export(output_file, format="wav")
  return output_file
converted_audio_file = convert_audio_for_model(file_path)

# uploaded = wave.open(file_path, "r")
# filen = list(uploaded.keys())[0]
Fs = 44100
audio1 = ess.MonoLoader(sampleRate=Fs, filename=converted_audio_file)()
ipd.Audio(audio1, rate=Fs)
W = 1024
H = 512
w = ess.Windowing(type='hann')
fft = ess.FFT()
c2p = ess.CartesianToPolar()
pool = essentia.Pool()
acf = ess.AutoCorrelation()
for frame in ess.FrameGenerator(audio1, frameSize=W, hopSize=H):
    mag, phase, = c2p(fft(w(frame)))
    autocorr = acf(w(frame))
    pool.add('mag_spec', mag)
    pool.add('phase_spec', phase)
    pool.add('acf', autocorr)
freq = np.arange((int)(W/2) + 1)*(Fs/W)
# PYin using essentia
pyin = ess.PitchYinProbabilistic(
    frameSize=W, hopSize=H, lowRMSThreshold=0.0001, preciseTime=True)
pyin_p, vp = pyin(audio1)

sorted_pitch = []
for pitch in pyin_p:
    if (pitch > 0):
        sorted_pitch.append(pitch)

# print(sorted_pitch)
# print(len(sorted_pitch))

# convert hz to notes


def quantize_predictions(ideal_offset, freq):
    h = round(12 * math.log2(freq / C0))
    # test cases outside
    # h = round(12 * math.log2(freq / C0) - ideal_offset)
    octave = h // 12
    n = h % 12
    note = note_names[n] + str(octave)
    return note


A4 = 440
C0 = A4 * pow(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
note_list = []
midi_notes = []


def cleanPitches(note_midi_nums, intervals):

    for i in range(5):
        note_midi_nums.append(-1)
    for i in range(5):
        note_midi_nums.insert(0, -1)
#   print(note_midi_nums)
#   print(len(note_midi_nums))
#   print(len(intervals))
    onset = (abs(intervals) >= 2) & (abs(intervals) <= 10)
    # print("onset", onset)
    window_size = 4  # 4 each side
    for i in range(5, len(note_midi_nums)-5):
        first_note = i-5
        last_note = i+5
        avg = 0
        c = 0
        # cumsum
        # make a temp array of 0s and 1s indicating which parts of the window we want to include in the window based on onset
        new_arr1 = onset[first_note:last_note]
        new_onset = 0
        j = i
        while j >= 0 and j < new_arr1.size and not new_arr1[j] == True:
            j -= 1
        if not j == 0:
            new_onset = j+1
            while j < new_arr1.size and not new_arr1[j] == True:
                j += 1
        end_onset = j
        start_onset = new_onset

        c = 0
        avg = 0
        for j in range(start_onset, end_onset):
            # DO STUFF
            c += 1
            avg += note_midi_nums[j]

        if not c == 0:
            avg = avg/c
        for j in range(start_onset, end_onset):
            note_midi_nums[j] = avg

        # for j in range(first_note, last_note):

        #   if (j < len(note_midi_nums) and (not note_midi_nums[j] == -1) and not abs(onset[j]) == 1):
        #     c+=1
        #     avg += note_midi_nums[j]
        # if (not c == 0):
        #   avg = int(avg/c)
        # if (i < len(note_midi_nums)-1 and (avg == note_midi_nums[i+1] or intervals[i] == -1 or intervals[i] == 1 or intervals[i] > 10 or intervals[i] < -10)):
        #     del note_midi_nums[i]
        #     del note_midi_nums[i]
        # elif (i < len(note_midi_nums) and (avg == note_midi_nums[i] or avg == note_midi_nums[i-1])):
        #   note_midi_nums[i] = avg

    # TODO: clean more than an octave jumps
    i = 0
    tot_avg = 0

    while i < len(note_midi_nums):
        if note_midi_nums[i] < 35:
            del note_midi_nums[i]
        else:
            tot_avg += note_midi_nums[i]
            i += 1
    tot_avg = tot_avg/len(note_midi_nums)
    i = 0
    while i < len(note_midi_nums):
        if note_midi_nums[i] < (tot_avg-6):
            del note_midi_nums[i]
        else:
            i += 1
# print(note_midi_nums)
    return note_midi_nums


def getNotesFromArray(uploaded_file_name):
    pitch_outputs = []
    for i in range(len(uploaded_file_name)):
        pitch_outputs.append(uploaded_file_name[i])
    # offsets = [hz2offset(p) for p in pitch_outputs if p != 0]
    # print("offsets: ", offsets)
    # ideal_offset = statistics.mean(offsets)
    # print("ideal offset: ", ideal_offset)
    for i in range(len(pitch_outputs)):
        ideal_offset = 0
        note_list.append(quantize_predictions(ideal_offset, pitch_outputs[i]))
        # note_list.append(quantize_predictions(ideal_offset, pitch_outputs[i]))
    i = 1
    # print(note_list)
    note_midi_nums = []
    new_note_names = ["A", "A#", "B", "C", "C#",
                      "D", "D#", "E", "F", "F#", "G", "G#"]
    for x in range(len(note_list)):
        temp_list = [*note_list[x]]
        i = 0
        if (len(temp_list)) == 2:
            i = new_note_names.index(temp_list[0])
        else:
            s = temp_list[0] + temp_list[1]
            i = new_note_names.index(s)
        note_midi_nums.append((21 + (int(temp_list[-1])-1)*12 + i))

    # while i < len(note_list):
    #   if (note_list[i] == note_list[i-1]):
    #     del note_list[i]
    #   else:
    #     i+=1
    # print(midi_notes)
    return note_midi_nums
    # print(note_list)


midi_notes = getNotesFromArray(sorted_pitch)
temp_arr = np.array(midi_notes)
new_arr = np.concatenate((np.array([0]), temp_arr))
interval_arr = new_arr[:-1]-temp_arr
for i in range(len(interval_arr)):
    if (interval_arr[i] > 10 or interval_arr[i] < -10):
        interval_arr[i] = 0
# print("intervals ", interval_arr)
interval_arr = np.concatenate((np.array([0, 0, 0, 0, 0]), interval_arr))
interval_arr = np.concatenate((interval_arr, np.array([0, 0, 0, 0, 0])))
interval_arr[interval_arr == -1] = 0
interval_arr[interval_arr == 1] = 0
cleaned_midi_notes = cleanPitches(midi_notes, interval_arr)
print(cleaned_midi_notes)
file = open("output.txt","w") 
 
file.write(str(cleaned_midi_notes))
 
file.close() 