# DSPRO2-kinship_prediction

### Getting started

If you use VS Code, run it in the integrated terminal:
1. Create conda env: `conda create -n tf23 python=3.7 tensorflow=2.3`
2. Activate conda env `conda activate tf23` 
3. run `conda install -n tf23 ipykernel --update-deps --force-reinstall` 
4. Install requirements `pip install -r requirements.txt`

### Build Docker
1. run `docker build -t dspro2-frontend .`
2. run `docker run --name dspro2-frontend -p 8501:8501 dspro2-frontend`
3. open http://localhost:8501
