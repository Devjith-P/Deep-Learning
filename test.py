import subprocess
import os
import cv2
import numpy as np
# --------- PATHS (EDIT ONLY IF NEEDED) ----------
PIPER_BIN = "/home/devjith/Luminar/Deep Learning/piper/piper/piper"
VOICE_MODEL = "/home/devjith/Luminar/Deep Learning/piper/voices/en_US-hfc_female-medium.onnx"
OUT_WAV = "/tmp/piper_test.wav"
# -----------------------------------------------

def speak(text):
    # Run piper
    process = subprocess.Popen(
        [PIPER_BIN, "--model", VOICE_MODEL,"--length_scale","1.6", "--output_file", OUT_WAV],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    process.stdin.write(text.encode("utf-8"))
    process.stdin.close()
    process.wait()

    # Play audio
    subprocess.run(["aplay", OUT_WAV])



import time
import random





interview = {
    "intro":['hey','hello','hi'],
    'outro':['bye','tata','ok']
}

speak('Please    introduce yourself.')
