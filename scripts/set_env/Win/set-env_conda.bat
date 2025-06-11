@Echo off

rem Define here the path to your conda installation
set CONDAPATH=C:\\Users\\User\\miniconda3\\

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat

rem 使用 python 3.10 創環境，如我沒有 python 請先自行安裝
call conda create --name benchmark python=3.10 -y

@REM 進入環境
call conda activate benchmark

Echo & echo."[Install packages in requirements.txt] ======================================="
python -m pip install --upgrade pip

@REM 安裝套件
pip install -r requirements.txt

@REM 安裝 pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
@REM 自行安裝 cuda https://developer.nvidia.com/cuda-11-7-0-download-archive
exit