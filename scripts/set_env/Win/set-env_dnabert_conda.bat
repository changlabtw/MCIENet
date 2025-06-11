@Echo off

rem Define here the path to your conda installation
set CONDAPATH=C:\\Users\\User\\miniconda3\\

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat

rem 使用 python 3.10 創環境，如我沒有 python 請先自行安裝
call conda create --name dnabert python=3.8 -y

@REM 進入環境
call conda activate dnabert

Echo & echo."[Install packages in requirements.txt] ======================================="
python -m pip install --upgrade pip

@REM 安裝套件
@REM pip install triton==2.0.0.dev20221202
@REM pip install torch==1.13.1
pip install einops==0.6.1
pip install peft==0.4.0
pip install huggingface-hub==0.16.4
pip install scikit-learn
pip install matplotlib
pip install progressbar
pip install tensorboard==2.13.0
pip install tensorboard-data-server==0.7.1

@REM 安裝 pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
exit