# opticalFlake 

## Install
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r pip_requirements.txt

## Run
source .venv/bin/activate
python opticalFlake_V0.3.py

## Dependency File Update
pip freeze > pip_requirements.txt
