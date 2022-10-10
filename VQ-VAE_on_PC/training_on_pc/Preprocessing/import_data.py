import numpy as np
from os import listdir
from os.path import join, dirname
# import json
# from sklearn.utils import shuffle


def import_data(category, frame=32, channels=32, anomaly=True):
    # path to data
    root = dirname(dirname(__file__))  # noqa: E501
    time = '2s'
    frame = str(frame) + '_frame'
    band = str(channels) + '_band'
    save_path = join(root, 'Data', category)
    print(save_path)

    # # category = path.split('/')[-1]
    data_type = f'{category}_{time}_{band}x{frame}'
    # with open(save_path + f'/feature_id_{time}_{band}x{frame}.json', 'r') as file:   # noqa: E501
    #     data_id = json.load(file)

    # get the id of dataset
    # id = category.split('/')[-1] if '/' in category else category
    # id_anomaly = data_id[id+'_anomaly']
    # id_normal = data_id[id+'_normal']

    normal_feat_path = join(save_path, 'normal')
    anomaly_feat_path = join(save_path, 'anomaly')

    # load features
    norm_file_list = listdir(normal_feat_path)
    norm_file_list = [join(normal_feat_path, file) for file in norm_file_list]
    normal_feature = []

    for file in norm_file_list:
        a = np.load(file)['arr_0']
        normal_feature.append(a)

    normal_feature = np.concatenate(normal_feature, axis=0)

    # rescale data feature
    MAX = np.max(normal_feature)
    MIN = np.min(normal_feature)
    norm_normal_feature = np.clip((normal_feature-MIN)/(MAX-MIN), a_min=0, a_max=1)   # noqa: E501

    # reshape
    norm_normal = np.reshape(norm_normal_feature, (-1, norm_normal_feature.shape[1], norm_normal_feature.shape[2], 1))   # noqa: E501

    # divide data into different sets
    train_data = 0.8

    data_variance = np.var(norm_normal)
    np.random.shuffle(norm_normal)
    # norm_normal, id_normal = shuffle(norm_normal, id_normal)
    a = norm_normal.shape[0]
    print(f"Normal set size: {a}")
    train_normal_set = norm_normal[0:int(train_data*a)]
    # train_id = id_normal[0:int(train_data*a)]
    test_set = norm_normal[int(train_data*a):]
    # test_id = id_normal[int(train_data*a):]

    if anomaly:
        # get anomaly data
        abnorm_file_list = listdir(anomaly_feat_path)
        abnorm_file_list = [join(anomaly_feat_path, file) for file in abnorm_file_list]   # noqa: E501
        anomaly_feature = []

        for file in abnorm_file_list:
            a = np.load(file)['arr_0']
            anomaly_feature.append(a)

        anomaly_feature = np.concatenate(anomaly_feature, axis=0)

        norm_anomaly_feature = np.clip((anomaly_feature-MIN)/(MAX-MIN), a_min=0, a_max=1)   # noqa: E501
        norm_anomaly = np.reshape(norm_anomaly_feature, (-1, norm_anomaly_feature.shape[1], norm_anomaly_feature.shape[2], 1))   # noqa: E501

        b = norm_anomaly.shape[0]
        print(f'Anomaly set size: {b}')
        anomaly_set = norm_anomaly
    else:
        anomaly_set = []

    return train_normal_set, test_set, anomaly_set, data_variance, data_type, MAX, MIN # train_id, test_id, id_anomaly   # noqa: E501
