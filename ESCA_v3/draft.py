from config import autocfg
from pydub import AudioSegment
from pydub.utils import make_chunks

path = '/home/machida/Desktop/nqthinh/Data/park/Target3/splited(1116_4)037.wav'
audio = AudioSegment.from_file(path, "wav")
chunks = make_chunks(audio, 2000)


for index, item in enumerate(chunks):
    print(index)
    print(item)