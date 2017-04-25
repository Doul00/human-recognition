#! /usr/bin/env python3

import sys

import hog

if __name__ == '__main__':
    hog.get_hog_features(sys.argv[1], verbose=True)
