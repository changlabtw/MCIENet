@Echo off

rem Define here the path to your conda installation
set CONDAPATH=C:\Users\aaron\miniconda3

rem Check if conda directory exists
if not exist "%CONDAPATH%\" (
    echo Error: Conda installation not found at %CONDAPATH%
    echo Please modify the CONDAPATH in this script to point to your conda installation directory.
    echo Common locations include:
    echo   - C:\Users\%%USERNAME%%\miniconda3
    echo   - C:\Users\%%USERNAME%%\Anaconda3
    echo   - C:\ProgramData\Miniconda3
    echo   - C:\ProgramData\Anaconda3
    echo.
    echo To find your conda path, you can run: where conda
    pause
    exit /b 1
)

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call "%CONDAPATH%\Scripts\activate.bat"

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