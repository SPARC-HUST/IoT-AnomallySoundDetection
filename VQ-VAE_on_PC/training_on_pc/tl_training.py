from tensorflow.keras import optimizers
from Preprocessing import import_data
from Postprocessing import Postprocessing
from Trainer import ModelTrainer
from os.path import join, isdir, dirname
from os import mkdir
import numpy as np
import time


def tl_training(env=None, epoch=60, evaluate=True):
    # gate keeping check
    if not env:
        raise ValueError('Please specify the environment that will be trained.')

    root = dirname(__file__)
    path_to_save_model = join(root,'Results/Saved_models')
    path_to_save_graphs = join(root, 'Results/Graphs')
    sub = 'tl'

    # features characteristic: (bandsxframes) images, model's name
    frames = [32]
    bands = [32]
    models = ['vq_vae']

    # change the below code block if adding more environment
    if env=='intersection':
        categories = ['intersection']
        sources = {
            'intersection' : ['intersection', True],
            'Target3' : ['target/intersection/Target1', False],
            'Target2' : ['target/intersection/Target2', True],
        }
        targets = {
            'Target1': ['target/intersection/Target1', True],
        }
    elif env=='park':
        categories = ['park']
        sources = {
            'park' : ['park', True],
            'Target1' : ['target/park/Target1', False],
            'Target2' : ['target/park/Target2', False],
        }
        targets = {
            'Target3': ['target/park/Target3', True]
        } # noqa: E501
    else:
        raise ValueError(f'The {env} environment has not been implemented')

    # load in the data and start training
    for target, attr in targets.items():
        for frame, band in zip(frames, bands):
            # load source data
            s_train_normal_set = []
            s_anomaly_set = []
            s_data_type = []
            for source in sources:
                print(f'Loading source: {source}')
                s_normal, _, s_anomaly, _, s_type, _, _ = import_data(sources[source][0], frame, band, sources[source][1])   # noqa: E501
                s_train_normal_set.append(s_normal)
                # check if there are abnormal data for the current set
                if type(s_anomaly) == np.ndarray:
                    s_anomaly_set.append(s_anomaly)
                s_data_type.append(s_type)
            s_train_normal_set = np.concatenate(s_train_normal_set, axis=0)
            s_anomaly_set = s_anomaly_set[0] if len(s_anomaly_set)==1 else np.concatenate(s_anomaly_set, axis=0)
            s_data_type = s_data_type[-1]

            # load target data
            t_train_normal_set, t_test_set, t_anomaly_set, _, t_data_type, max, min = import_data(attr[0], frame, band, attr[1])   # noqa: E501
            print(f'Loading target: {t_data_type}')

            # create some data set
            train_normal_set = np.concatenate((s_train_normal_set, t_train_normal_set))   # noqa: E501
            train_variance = np.var(train_normal_set)
            test_set = t_test_set
            anomaly_set = t_anomaly_set
            training_anomaly = s_anomaly_set
            data_type = t_data_type

            print(f'Number of train data: {train_normal_set.shape[0]}')
            print(f'Number of test data: {test_set.shape[0]}')
            print(f'Number of training anomaly data: {training_anomaly.shape[0]}')    # noqa: E501

            # need to get original shape of vae set for evaluation
            og_test_shape = test_set.shape
            print(f'Actual normal test samples: {og_test_shape}')
            og_anomaly_shape = anomaly_set.shape if type(anomaly_set)==np.ndarray else 0
            print(f'Actual anomaly test samples: {og_anomaly_shape}')

            # initiate the model and some important constants
            epochs = epoch
            BATCH_SIZE = 256 if train_normal_set.shape[0] > 2000 else 64
            STEPS_PER_EPOCH = train_normal_set.shape[0]//BATCH_SIZE

            for name in models:
                print(f'Training {name}')
                if not 'Target' in s_data_type:
                    base = join(path_to_save_model, name, 'source', s_data_type)
                else:
                    base = join(path_to_save_model, name, sub ,s_data_type)

                # if using the Trainer for transfer transfer learning
                # the key words parameters needed to pass in are:
                # beta: the param to control how much supervised loss contribute to the total loss   # noqa: E501
                # training_anomaly: the anomaly set of source set for training
                # batch_size: BATCH_SIZE
                tl = True
                trainer = ModelTrainer(name, train_variance, train_normal_set.shape[1]*4, transfer_learning=tl,   # noqa: E501
                                       beta=1.0, training_anomaly=training_anomaly, batch_size=BATCH_SIZE,
                                       based_model=base, step_num=STEPS_PER_EPOCH)   # noqa: E501
                trainer.compile(optimizer=optimizers.Adam())

                # adding start time and end time to calculate training time
                t_start = time.time()

                history = trainer.fit(train_normal_set,
                                      steps_per_epoch=STEPS_PER_EPOCH,
                                      epochs=epochs)

                t_end = time.time()
                print(f"Training time: {(t_end-t_start)/60} mins.")

                if evaluate:
                    results = Postprocessing(trainer.model, name, history, test_set,
                                              anomaly_set, data_type, og_test_shape,
                                              og_anomaly_shape, max, min)    # noqa: E501
                    # saving metrics
                    if type(t_anomaly_set) == np.ndarray:
                        graph_path = join(path_to_save_graphs, name, sub, data_type)
                        results.save_results(graph_path)

                # saving model
                model_path = join(path_to_save_model, name, sub, data_type)   # noqa: E501
                if not isdir(model_path):
                    mkdir(model_path)
                trainer.model.save(model_path)

                print(f'The training for {target} has finished.')
                del results
                print('-'*50)

            # release memory
            del train_normal_set
            del test_set
            del anomaly_set
    return 0
