@echo on

python --version > temp.txt 2>nul
findstr "3.12" temp.txt >nul
if %ERRORLEVEL% EQU 0 (
	echo Python 3.12.x is installed.
) else (
	echo Python 3.12.x is not installed.
	exit /b
)
del temp.txt

python -m venv venv
pushd venv\Scripts
call activate.bat
popd
pip install -r requirements.txt