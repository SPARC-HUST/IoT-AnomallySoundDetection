from os.path import join
from Preprocessing import Extract_features


if __name__ == '__main__':

    feature = 'gamma'

    # path to the audio folder
    norm_audio = []
    abnorm_audio = []

    # path to save features, should be under the Data folder
    feature_folder = [f'/home/minh/Documents/ESCA/Data/44.1k_data/{feature}_feature/partition_feature']

    categories = ['intersection/Target1', 'intersection/Target2', 'intersection/Target3',    # noqa: E501
                  'park/Target1', 'park/Target2', 'park/Target3']
    length = 2

    # some parameters
    # 2s file: winddow_time = 0.06*2 for 32 frame -> nfft = 5292, .0305*2 for 64 frame    # noqa: E501
    window_time = [0.06*2]
    channels = [32]


    hop_time = [w/2 for w in window_time]
    fmin = 100
    f = 44100

    time = str(length)+'s'
    handler = Extract_features()

    for wt, ht, channel in zip(window_time, hop_time, channels):
        save_path = []
        for folder in feature_folder:
            frame = str(int((length-wt)//ht)+1)+'_frame'
            band = str(channel)+'_band'
            for category in categories:
                save_path.append(join(folder, time, frame, band, 'target', category))    # noqa: E501

        # print(save_path)

        norm_feature = [join(path, 'normal') for path in save_path]
        abnorm_feature = [join(path, 'anomaly') for path in save_path]

        for p, src1, dst1, src2, dst2 in zip(save_path, norm_audio, norm_feature, abnorm_audio, abnorm_feature):    # noqa: E501
            if feature == 'gamma':
                handler.gamma_features(src1, dst1, length, wt, ht, channel, fmin)    # noqa: E501
                handler.gamma_features(src2, dst2, length, wt, ht, channel, fmin)    # noqa: E501
            else:
                nfft_ = int((wt+wt/15)*f)
                # print(nfft_)
                handler.mel_features(src1, dst1, length, f, nfft_, channel)
                handler.mel_features(src2, dst2, length, f, nfft_, channel)

            path = p
            # category = path.split('/')[-1]
            name = f'/feature_id_{time}_{band}x{frame}.json'    # noqa: E501
            handler.save_id(path, name)
