from pydub import AudioSegment
from os.path import join, isdir
from os import listdir, remove, mkdir
from os.path import dirname


def clean_up(file_name):
    # making directory
    root = dirname(__file__)
    sub_dir = 'Data/history'
    folder = file_name.split('_')[0]
    path = join(root,sub_dir,folder)
    if not isdir(path):
        mkdir(path)

    # prepare the file
    audio_loc = join(root, 'test_samples/temp')
    audio_2s = sorted(listdir(audio_loc))
    audio_list = [join(audio_loc, a) for a in audio_2s]

    # save audio in to a big file and at the same time remove that file
    combined = AudioSegment.empty()
    for file in audio_list:
        audio = AudioSegment.from_file(file, "wav")
        combined += audio
        remove(file)

    combined.export(join(path, f'{file_name}.wav'), format="wav")

    print(f'The recorded audio can be found at {path}')
