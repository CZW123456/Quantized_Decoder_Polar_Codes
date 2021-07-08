clear
rm -rf dist
python3 setup.py bdist_wheel
cd dist
pip uninstall quantizers --yes
pip install *.whl --upgrade
