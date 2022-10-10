from os.path import join, isdir
from os import mkdir
from pydub import AudioSegment
from pydub.utils import make_chunks
from .utils import read_file_name


root = '/home/minh/Documents/ESCA'


# a function segment the audio file to desired length in second
def segmentation(src, dst, length=10):
    # src is the source folder
    # dst is the destination folder

    time_per_file = length*1000  # time is processed in millisecond

    src = join(root, src)
    if not isdir(src):
        return 1

    print(src)
    file_list = read_file_name(src)
    print(len(file_list))

    dst = join(root, dst)
    print(dst)
    if not isdir(dst):
        mkdir(dst)

    for file in file_list:
        audio = AudioSegment.from_file(file, "wav")
        chunks = make_chunks(audio, time_per_file)
        file_name = file.split('/')[-1]

        # write chunks to dst
        for index, item in enumerate(chunks):
            chunk_name = file_name[:-4] + '_' + str(index) + '.wav'
            print("exporting ", chunk_name)
            item.export(join(dst, chunk_name), format='wav')

    return 0
