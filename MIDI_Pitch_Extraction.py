from music21 import *
def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

    return midi.translate.midiFileToStream(mf)

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        aux = p
        print (p.partName)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)

    return ret #ret,parent_element #midi val and note 
#converts midi input into intervals
def songProcessing(filepath):
  base_midi = open_midi(filepath, True)
  song = extract_notes(base_midi)
  prev_note = 0
  song_norep = []
  return song

 # print(len(song))
  #print(song)
  """ for i in range(len(song)):
      if song[i] != song[i-1]:
        song_norep.append(song[i])
  note_midi_intervals = []
  for i in range(len(song_norep)-1):
    note_midi_intervals.append(int(song_norep[i + 1] - song_norep[i]))
  i = 0
  while i < len(note_midi_intervals):
    if (note_midi_intervals[i] == -1 or note_midi_intervals[i] == 1):
      del note_midi_intervals[i]
    else:
      i+=1
  return (note_midi_intervals)
 """

