@Echo off

:start
@REM 使用 python 3.10 創環境，如我沒有 python 請先自行安裝
py -3.10 -m venv .venv && (
  Echo "Create the environment under the .venv folder"
) || (
  Echo "Python 3.10 not found. Please install it yourself first"
  exit /b 1
)

@REM 進入環境
call ".venv\Scripts\activate.bat"

Echo & echo."[Install packages in requirements.txt] ======================================="
python -m pip install --upgrade pip

@REM 安裝套件
pip install -r requirements.txt
Echo & echo."[Install Pytorch] ============================================================"

@REM 安裝 pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pause
exit