#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from opt import opt
from pathlib import Path

LOG_PATH = None if opt.log is None else Path(opt.log)
LOG_FORMAT = '[%(asctime)s] [%(filename)s@%(lineno)d] [%(levelname)s] %(message)s'

logging.basicConfig(level=logging.DEBUG, filename=LOG_PATH, format=LOG_FORMAT)

if __name__ == '__main__':
    logging.debug('Hello world!')
