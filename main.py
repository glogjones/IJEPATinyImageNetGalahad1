# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import pprint
import yaml
import torch

from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which device to use on local machine (default is cuda:0 for single GPU)')

def process_main(rank, fname, devices):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[0].split(':')[-1])  # Use the first device specified

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f'called-params {fname}')

    # Load script parameters
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    logger.info(f'Running on device: {devices[0]}')
    app_main(args=params)

if __name__ == '__main__':
    args = parser.parse_args()

    # Run process_main directly for single-GPU execution
    process_main(0, args.fname, args.devices)
