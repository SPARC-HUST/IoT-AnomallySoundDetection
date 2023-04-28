
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from pydub import AudioSegment

file = AudioSegment.from_file("./Data/raw/Target3/abnormal/splited(1116_4)006_0.wav", "wav")
value= tf.io.serialize_tensor(file)

print(value)