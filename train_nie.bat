@echo off
echo Starting training...
python -u -m nova.nie_trainer > training_output.txt 2>&1
echo Training finished.
