@echo off
REM call python directly with full path; script in same folder as the bat
"C:\Users\kaden.mcculloch\AppData\Local\anaconda3\envs\kfmcculloch98\python.exe" "%~dp0run_theis_forward_heads.py" proj6.yml
exit /b %errorlevel%