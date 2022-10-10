from tensorflow.keras import optimizers
from Preprocessing import import_data
from Postprocessing import Postprocessing
from Trainer import ModelTrainer
from os.path import join, isdir, dirname
from os import mkdir
import numpy as np
import time


def base_training(env=None, target=None, epoch=300, evaluate=True, anom=True):
    # gate keeping check
    if not env:
        raise ValueError('Please specify the environment that will be trained.')
    if not target:
        raise ValueError('Please specify a target for evaluating.')

    root = dirname(__file__)
    path_to_save_model = join(root,'Results/Saved_models')
    path_to_save_graphs = join(root, 'Results/Graphs')
    sub = 'base' if target != 'source' else target

    # features characteristic: (bandsxframes) images, model's name
    frames = [32]
    bands = [32]
    models = ['vq_vae']

    tar = env + '/' + target if target != 'source' else env
    # change the below code block if adding more environment
    if env=='intersection':
        categories = ['intersection']
        targets = [tar] # noqa: E501
    elif env=='park':
        categories = ['park']
        targets = [tar] # noqa: E501
    else:
        raise ValueError(f'The {env} has not been implemented')

    # load in the data and start training
    for category, target in zip(categories, targets):
        for frame, band in zip(frames, bands):

            # load target data
            n = 'target/'+target if sub != 'source' else target
            t_train_normal_set, t_test_set, t_anomaly_set, _, t_data_type, max, min = import_data(n, frame, band, anom)   # noqa: E501
            print(f'Loading target: {t_data_type}')

            # create some data set
            train_normal_set = t_train_normal_set   # noqa: E501
            train_variance = np.var(train_normal_set)
            test_set = t_test_set
            anomaly_set = t_anomaly_set
            data_type = t_data_type

            print(f'Number of train data: {train_normal_set.shape[0]}')
            print(f'Number of test data: {test_set.shape[0]}')

            # need to get original shape of vae set for evaluation
            og_test_shape = test_set.shape
            # print(f'Actual normal test samples: {og_test_shape}')
            og_anomaly_shape = anomaly_set.shape if type(anomaly_set)==np.ndarray else 0
            # print(f'Actual anomaly test samples: {og_anomaly_shape}')

            # initiate the model and some important constants
            epochs = epoch
            BATCH_SIZE = 256 if train_normal_set.shape[0] > 2000 else 64
            STEPS_PER_EPOCH = train_normal_set.shape[0]//BATCH_SIZE

            for name in models:
                print(f'Training {name}')

                tl = False
                trainer = ModelTrainer(name, train_variance, train_normal_set.shape[1]*4, transfer_learning=tl)   # noqa: E501
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
                model_path = join(path_to_save_model, name, sub, data_type)    # noqa: E501
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
