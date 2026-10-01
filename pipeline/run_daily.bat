@echo off
REM Daily hydro pipeline: SWE model -> swe_daily, Energy Quantified river temperatures -> river_temp_eq.
REM Runs run_daily.py in the snow_obs conda environment (the one 102_SWE_lorenzo.bat uses, which also
REM has the Energy Quantified client and the Databricks SQL connector). Any argument is passed through,
REM e.g.  run_daily.bat --skip-swe-model   or   run_daily.bat --backfill
REM
REM Task Scheduler: Program  = cmd.exe
REM                 Arguments = /c "P:\QFA\TonyWeather\Power_dashboard\pipeline\run_daily.bat"
REM                 Trigger   = daily 11:30 (Exolabs publishes ~10:45; SWE_main.py uses yesterday before that)
setlocal
set "MINIFORGE=%LOCALAPPDATA%\miniforge3"
if not exist "%MINIFORGE%\Scripts\activate.bat" set "MINIFORGE=C:\ProgramData\anaconda3"
set PYTHONIOENCODING=utf-8
echo [%date% %time%] hydro pipeline starting in snow_obs ...
if exist "%MINIFORGE%\Scripts\activate.bat" (
    call "%MINIFORGE%\Scripts\activate.bat" snow_obs
    python "%~dp0run_daily.py" %*
) else (
    call conda run -n snow_obs python "%~dp0run_daily.py" %*
)
set RC=%ERRORLEVEL%
echo [%date% %time%] finished with exit code %RC%
endlocal & exit /b %RC%
