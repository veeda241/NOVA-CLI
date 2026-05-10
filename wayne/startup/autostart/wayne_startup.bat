@echo off
title W.A.Y.N.E DAEMON
cd /d "C:\path\to\wayne"
echo Starting W.A.Y.N.E background services...
python startup\wayne_daemon.py
pause
