from tensorflow.keras.models import load_model
from Preprocessing import import_data
from Postprocessing import Postprocessing
from os.path import join, isdir, dirname, exists
import numpy as np
import json


def evaluating(env=None, target=None, transfer_learning=True, clean_up_json=None, anom=True):
    # gate keeping check
    if not env:
        raise ValueError('Please specify a enviromnet for evaluating.')
    if not target:
        raise ValueError('Please specify a target for evaluating.')
    if not anom:
        raise ValueError('Can not perform evaluation if there is no anomaly sample.')

    root = dirname(__file__)
    type = 'tl' if transfer_learning else 'base'
    type = 'source' if target == 'source' else type
    path_to_load_model = join(root, 'Results/Saved_models')
    path_to_save_graphs = join(root, 'Results/Updated_Graphs')


    categories = [env]
    frames = [32]
    bands = [32]
    models = ['vq_vae']
    targets = [env + '/' + target] if target != 'source' else [env]

    # load the clean_up.json file for actual anomaly samples if specified
    if clean_up_json:
        clean_up_file_location = join(root, clean_up_json)   # noqa: E501
        if not exists(clean_up_file_location):
            raise ValueError(f'{clean_up_json} does not exist.')
        with open(clean_up_file_location, 'r') as file:
            clean_up = json.load(file)

    for category, target in zip(categories, targets):
        for frame, band in zip(frames, bands):
            # load source data
            n = 'target/'+target if type != 'source' else target
            t_train_normal_set, t_test_set, t_anomaly_set, _, t_data_type, max, min = import_data(n, frame, band)   # noqa: E501
            print(f'Loading target: {t_data_type}')

            # create some data set
            test_set = t_test_set
            anomaly_set = t_anomaly_set
            data_type = t_data_type

            target_num = target.split('/')[-1]
            if clean_up_json:
                if type != 'source':
                    anomaly_set = np.array([anomaly_set[i] for i in clean_up[category][target_num]])   # noqa: E501
                else:
                    anomaly_set = np.array([anomaly_set[i] for i in clean_up[category]])
            print(f'After cleaning up: {anomaly_set.shape[0]}')
            print(f'Loading data: {data_type}')
            # need to get original shape of vae set for evaluation
            og_test_shape = test_set.shape
            print(f'Number of normal samples: {og_test_shape}')
            og_anomaly_shape = anomaly_set.shape
            print(f'Number of anomaly samples: {og_anomaly_shape}')

            for name in models:
                print(f'Evaluating {name} with {target}')
                # Loading model
                # data_type1 = f'intersection_2s_{band}_bandx{frame}_frame'
                model_path = join(path_to_load_model, name, type, data_type)
                if not isdir(model_path):
                    raise ValueError(f'The {model_path} is not valid.')
                # Save trained model
                model = load_model(model_path)

                if name == 'vae':
                    # vae need a different input so need to reshape
                    vae_test_set = np.reshape(np.transpose(test_set, (0, 2, 1, 3)), (-1, test_set.shape[1]*4))   # noqa: E501
                    vae_anomaly_set = np.reshape(np.transpose(anomaly_set, (0, 2, 1, 3)), (-1, anomaly_set.shape[1]*4))   # noqa: E501

                    results = Postprocessing(model, name, None, vae_test_set, vae_anomaly_set,
                                             data_type, og_test_shape, og_anomaly_shape, max, min)    # noqa: E501
                else:
                    results = Postprocessing(model, name, None, test_set, anomaly_set,
                                             data_type, og_test_shape, og_anomaly_shape, max, min)    # noqa: E501

                # saving metrics
                graph_path = join(path_to_save_graphs, name, type, data_type)
                results.save_results(graph_path)

                del results
                print(f'The evaluating for {target} has finished.')
                print('-'*50)

            # release memory
            del test_set
            del anomaly_set
    return 0
