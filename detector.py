#! /usr/bin/env python3

import sys

import tirf

if __name__ == '__main__':
    tirf.get_hog_features(sys.argv[1], verbose=True)
