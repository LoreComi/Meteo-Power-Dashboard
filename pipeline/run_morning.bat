@echo off
REM Morning Call + Gas Demand feed from Volue (wapi) -> morning_daily_volue. (The EQ feed is: run_daily.py --only morning.)
REM Schedule this OFTEN (Task Scheduler, e.g. daily at 02:30 then repeat every 2 hours for 22 hours):
REM each run picks up the issues of the last 36 hours, so a new 00/06/12/18 cycle is in the
REM dashboard on the first run after EQ publishes it, and the agents brief on the latest one.
REM The 08:00 run has every model's 00z; the daily SWE/river job (run_daily.bat) also includes this step.
setlocal
set "MINIFORGE=%LOCALAPPDATA%\miniforge3"
if not exist "%MINIFORGE%\Scripts\activate.bat" set "MINIFORGE=C:\ProgramData\anaconda3"
set PYTHONIOENCODING=utf-8
echo [%date% %time%] morning call feed starting in snow_obs ...
if exist "%MINIFORGE%\Scripts\activate.bat" (
    call "%MINIFORGE%\Scripts\activate.bat" snow_obs
    python "%~dp0run_daily.py" --only volue %*
) else (
    call conda run -n snow_obs python "%~dp0run_daily.py" --only volue %*
)
set RC=%ERRORLEVEL%
echo [%date% %time%] finished with exit code %RC%
endlocal & exit /b %RC%
