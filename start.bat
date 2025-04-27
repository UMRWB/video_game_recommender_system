@echo on
pushd venv\Scripts
call activate.bat
popd
call streamlit run app.py