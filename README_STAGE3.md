# Stage 3 incremental patch for Stage 1 + Stage 2 project

Extract this patch **over** `C:\Users\ASUS\mlhat-reproduction\mlhat_term_project_starter`, using the existing virtual environment. It only adds files and does not replace Stage 1/2 files.

```powershell
# Run in the project root containing src, experiments and tests
Expand-Archive -Path "$HOME\Downloads\mlhat_stage3_patch.zip" -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
python .\experiments\run_stage3.py --model stage2 --samples 3000 --seed 42
python .\experiments\run_stage3.py --model stage3 --samples 3000 --seed 42
```

Read `docs/stage3_notes.md` for the exact scope and limitations. This is a *technical learning prototype*, **NOT a complete or faithful MLHAT reproduction**.
