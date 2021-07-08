clear
rm -rf dist
python3 setup.py bdist_wheel
cd dist
pip uninstall libPolarDecoder --yes
pip install *.whl -U --upgrade yes

