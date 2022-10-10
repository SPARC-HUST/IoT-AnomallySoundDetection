import wave
import datetime
import time
import os
import shutil
import subprocess

def CPfile(fold_file):
    if (os.path.exists("./test_samples/test/basefile.wav") == False):
        shutil.copy2(fold_file, "./test_samples/test/basefile.wav")

if __name__=='__main__':
    channels = 1
    rate = 44100  # Record at 8000 samples per second
    seconds = 2
    try:
        i = 0
        while True:
            print("Start cre Audio file")

            try:
                os.system(f"arecord -D plughw:0,0 -f S16_LE -r {rate} -d {seconds}  ./test_samples/test/output.wav > /dev/null 2>&1")
            except:
                print("os.system() failed")
            print("End cre Audio file")

            CPfile("./test_samples/test/output.wav")
            if (os.path.exists("./test_samples/test/output.wav") == True):
                os.remove("./test_samples/test/output.wav")
    except KeyboardInterrupt as e:
        print(e)
