#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--log',
                        type=str,
                        default=None)
arg_parser.add_argument('--train',
                        type=str,
                        default='../../data/train.csv')
arg_parser.add_argument('--test',
                        type=str,
                        default='../../data/test.csv')

arg_parser.add_argument('--seed',
                        type=int,
                        default=100)

opt, _ = arg_parser.parse_known_args()
