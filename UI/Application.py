import threading
import time
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty
from kivy.properties import NumericProperty
import sounddevice as sd
from scipy.io.wavfile import write
from kivy.clock import Clock

# Sampling frequency
freq = 44100

#duration
duration = 15

class Final(FloatLayout):
    number = NumericProperty()
    major = StringProperty("")
    minor = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file = StringProperty("")

    def wait(self):
        time.sleep(15)
        self.stop()
        self.ids.Recording.pos_hint = {"center_x": .5, "center_y": 5000}
        self.major = "C"
        self.minor = "A"
        self.ids.watch.y = 1000
        self.ids.final_display.pos_hint = {"center_x": .5, "center_y": .5}
    
    def rec(self):
        self.recording = sd.rec(int(duration * freq), samplerate = freq, channels = 2)
        sd.wait()
        write("recording0.wav", freq, self.recording)

    def on_button_click(self):
        self.ids.title.pos_hint = {"center_x": .5, "center_y": 100}
        self.ids.but_input.y = 5000
        self.ids.but_record.y = 5000
        self.ids.stop_input.y = 0
        self.ids.input_instructions.pos_hint = {"center_x": .5, "center_y": .8}
        self.ids.input.y = 180

    def before_record(self):
        self.ids.title.pos_hint = {"center_x": .5, "center_y": 100}
        self.ids.Recording.pos_hint = {"center_x": .5, "center_y": .65}
        self.ids.but_input.y = 5000
        self.ids.but_record.y = 5000
        self.ids.watch.y = 0
        self.start_recording()
        self.start()
        
    def start_recording(self):
        self.recording_thread = threading.Thread(target = self.rec)
        self.recording_thread.start()
        self.waiting = threading.Thread(target = self.wait)
        self.waiting.start()
        
    def stop_input(self):
        self.ids.stop_input.y = 5000
        self.ids.input_instructions.pos_hint = {"center_x": .5, "center_y": 1000}
        self.ids.input.y = 20000
        self.major = "C"
        self.minor = "A"
        self.file = self.ids.input.text
        self.ids.final_display.pos_hint = {"center_x": .5, "center_y": .5}

    def increment_time(self, interval):
        self.number += .1
    
    def start(self):
        Clock.unschedule(self.increment_time)
        Clock.schedule_interval(self.increment_time, .1)
    
    def stop(self):
        Clock.unschedule(self.increment_time)
    

class HumWizApp(App):
    pass

HumWizApp().run()
