@echo off
REM Scenario inputs for the Forecast -> Scenarios tab, twice a day after the 00z and 12z runs:
REM   z500        EC-ENS 500 hPa geopotential members from the Meteomatics API -> fcst_member_spatial_ecens
REM   eq-members  AIFS-ENS per-member temperature / wind / solar by country from EQ -> fcst_members_eq
REM Task Scheduler: Program  = cmd.exe
REM                 Arguments = /c "P:\QFA\TonyWeather\Power_dashboard\pipeline\run_scenarios.bat"
REM                 Trigger   = daily 08:30 and 20:30 (EC-ENS 00z is complete in Meteomatics ~07:00 UTC, the 12z ~19:00 UTC;
REM                             a run already in the table is skipped, so an extra trigger costs nothing)
setlocal
set "MINIFORGE=%LOCALAPPDATA%\miniforge3"
if not exist "%MINIFORGE%\Scripts\activate.bat" set "MINIFORGE=C:\ProgramData\anaconda3"
set PYTHONIOENCODING=utf-8
echo [%date% %time%] scenario inputs starting in snow_obs ...
if exist "%MINIFORGE%\Scripts\activate.bat" (
    call "%MINIFORGE%\Scripts\activate.bat" snow_obs
    python "%~dp0run_daily.py" --only z500 --only eq-members %*
) else (
    call conda run -n snow_obs python "%~dp0run_daily.py" --only z500 --only eq-members %*
)
set RC=%ERRORLEVEL%
echo [%date% %time%] finished with exit code %RC%
endlocal & exit /b %RC%
