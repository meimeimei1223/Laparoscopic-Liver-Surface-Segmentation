#!/usr/bin/env bash
# Serve this static annotation app on http://localhost:8000 (Linux / macOS).
# Uses stock python3 -- no pip packages, no GPU. Use Chrome/Chromium for the
# "pick masks folder" direct-write (File System Access API).
cd "$(dirname "$0")"
echo "Opening http://localhost:8000  (Ctrl+C to stop)"
( command -v xdg-open >/dev/null && xdg-open http://localhost:8000 ) || \
( command -v open >/dev/null && open http://localhost:8000 ) || true
python3 -m http.server 8000 --bind 127.0.0.1
