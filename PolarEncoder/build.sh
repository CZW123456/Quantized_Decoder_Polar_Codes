#!/usr/bin/env bash
clear
rm -rf dist
python3 setup.py bdist_wheel
cd dist
pip uninstall PolarBD --yes
pip install *.whl --upgrade
