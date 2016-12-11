#!/bin/sh
#Script to run MS segmentation for a bunch of params
./mumfordshah.py butterfly.png -lambda 0.1
./mumfordshah.py butterfly.png -lambda 0.01
./mumfordshah.py butterfly.png -lambda 0.001
./mumfordshah.py butterfly.png -lambda 0.0001
./mumfordshah.py butterfly.png -lambda 0.00001
./mumfordshah.py butterfly.png -lambda 0.000001

