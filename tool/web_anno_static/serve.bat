@echo off
REM Serve this static annotation app on http://localhost:8000
REM Uses stock Python's built-in server -- NO conda env, NO packages, NO GPU.
REM (Serving on localhost is what lets the "pick masks folder" direct-write work.)
cd /d "%~dp0"
echo Opening http://localhost:8000  (close this window to stop)
start "" "http://localhost:8000"
python -m http.server 8000 --bind 127.0.0.1
pause
