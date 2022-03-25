# create venv
python3 -m venv ./venv
# activate venv
source ./venv/bin/activate
# build pyscf
python3 setup.py build
# install pyscf
python3 setup.py install
# run DFT numint test
python3 pyscf/dft/test/test_numint.py
# run benchmark
python3 pyscf_testbench.py
# deactive venv
deactivate