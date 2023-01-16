from argparse import ArgumentParser
import subprocess
from evaluating import evaluating
from b_training import base_training
from tl_training import tl_training
from inferencing import inferencing
from rt_test import testing
from audio_cleanup import clean_up
from os.path import dirname, abspath, join
import signal
from os import setpgrp, killpg, _exit
import json
import sys
from datetime import datetime


if __name__=='__main__':
    parser = ArgumentParser(description='program for running other processes')
    subparser = parser.add_subparsers(dest='command')

    b_train = subparser.add_parser('base_training')
    t_train = subparser.add_parser('tl_training')
    evaluate = subparser.add_parser('evaluating')
    infer = subparser.add_parser('inferencing')
    rtest = subparser.add_parser('test')

    # arguments that is needed for every type
    parser.add_argument('-env', '--environment', help='specify the env',
                        required=True, choices=['intersection', 'park'])
    parser.add_argument('-log', '--logPath', help='path to the log file')

    # arguments for transfer training only
    t_train.add_argument('-ep', '--epochs', help='training epochs', type=int)
    t_train.add_argument('-eval', '--evaluate', help='enable evaluation after training',
                        action='store_true')

    # arguments for base training only
    b_train.add_argument('-tar', '--target', help='specify the target for training process',
                          choices=['source', 'Target1', 'Target2', 'Target3'], required=True)
    b_train.add_argument('-ep', '--epochs', help='training epochs', type=int)
    b_train.add_argument('-eval', '--evaluate', help='enable evaluation after training',
                        action='store_true')
    b_train.add_argument('-ab', '--abnormal', help='whether data set contains abnormal data or not',
                        action='store_true')

    # arguments for evaluating only
    evaluate.add_argument('-tar', '--target', help='specify the target for evaluating process',
                          choices=['source', 'Target1', 'Target2', 'Target3'], required=True)
    evaluate.add_argument('-cl', '--cleanUpJson', help='path to the clean up json file')
    evaluate.add_argument('-tl', '--transferLearning', help='specify if the target was trained by transfer learning or not',
                          action='store_true')
    evaluate.add_argument('-ab', '--abnormal', help='whether data set contains abnormal data or not',
                        action='store_true')

    # arguments for inferencing only
    infer.add_argument('-tar', '--target', help='specify the target for evaluating process',
                          choices=['source', 'Target1', 'Target2', 'Target3'], required=True)
    infer.add_argument('-th', '--manualThreshold', help='manually select the threshold')
    infer.add_argument('-tl', '--transferLearning', help='specify if the target was trained by transfer learning or not',
                          action='store_true')

    # arguments for test only
    rtest.add_argument('-tar', '--target', help='specify the target for evaluating process',
                          choices=['source', 'Target1', 'Target2', 'Target3'], required=True)
    rtest.add_argument('-th', '--manualThreshold', help='manually select the threshold')
    rtest.add_argument('-tl', '--transferLearning', help='specify if the target was trained by transfer learning or not',
                          action='store_true')
    rtest.add_argument('-rt', '--runtime', help='specify how long the test will be running (default to 300s)',
                        type=int, default=300)

    args = parser.parse_args()

    # running monitoring in another subprocess
    working_dir = dirname(abspath(__file__))
    print(f'Working directory: {working_dir}')
    log_file = join(working_dir, "log_file.json") if not args.logPath else args.logPath
    file = join(working_dir, 'monitoring.py')
    command = ['gnome-terminal', '--disable-factory','--', 'python3', file, '-log', log_file]
    # monitor = subprocess.Popen(command, preexec_fn=setpgrp)

    if args.command=='tl_training':
        # checking for inputs argument, else replacing with default value
        ep = 60 if not args.epochs else args.epochs
        tl_training(env=args.environment, epoch=ep, evaluate=args.evaluate)
    elif args.command=='base_training':
        ep = 300 if not args.epochs else args.epochs
        base_training(env=args.environment, target=args.target, epoch=ep, evaluate=args.evaluate, anom=args.abnormal)
    elif args.command=='evaluating':
        transfer = args.transferLearning
        evaluating(env=args.environment, target=args.target, transfer_learning=transfer, clean_up_json=args.cleanUpJson, anom=args.abnormal)
    elif args.command=='inferencing':
        transfer = args.transferLearning
        inferencing(env=args.environment, target=args.target, transfer_learning=transfer, manual_threshold=args.manualThreshold)
    else:
        prediction_list = []
        transfer = args.transferLearning
        now = datetime.now()
        cur_time = now.strftime("%Y%m%d_%H%M%S")
        try:
            audio_record = subprocess.Popen(['gnome-terminal', '--disable-factory','--', 'python3', 'micUSB_cmd.py'],
                                            preexec_fn=setpgrp)
            save_file = testing(env=args.environment, target=args.target, transfer_learning=transfer, manual_threshold=args.manualThreshold,
                                rtime=args.runtime, eval=prediction_list)
            killpg(audio_record.pid, signal.SIGINT)
            print('Cleaning up...')
            clean_up(cur_time)
        except KeyboardInterrupt as e:
            print('Get interrupted by keyboard')
            print('Saving the results so far...')
            killpg(audio_record.pid, signal.SIGINT)
            print('Cleaning up...')
            clean_up(cur_time)
            # killpg(monitor.pid, signal.SIGINT)
            try:
                sys.exit(0)
            except SystemExit:
                _exit(0)

    # killpg(monitor.pid, signal.SIGINT)
