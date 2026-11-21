MLHAT_17 — LAB1 PARTIAL REPRODUCTION PATCH

Extract this ZIP into your EXISTING mlhat_term_project_starter folder.
It only adds new benchmark scripts, tests, docs and a data README.
No Stage 1–3 code is overwritten.

1. With activated .venv, run:
   python -m pip install -e .
   python -m pip install liac-arff wandb
   python -m unittest discover -s tests -v

2. Download the complete Emotions dataset (Mulan .arff + .xml) from:
   https://www.uco.es/kdis/mllresources/
   Save as data/emotions.arff and data/emotions.xml.

3. Use: 
   python .\experiments\run_paper_benchmark.py --dataset Emotions --arff .\data\emotions.arff --xml .\data\emotions.xml --model stage3 --seed 42

4. For W&B:
   wandb login
   python .\experiments\run_paper_benchmark.py --dataset Emotions --arff .\data\emotions.arff --xml .\data\emotions.xml --model stage3 --seed 42 --wandb --project mlhat17-reproduction

5. Repeated model baseline/seed runs:
   .\experiments\run_lab1.ps1 -Dataset Emotions -Arff .\data\emotions.arff -Xml .\data\emotions.xml -Track

6. Read docs/LAB1_REPRODUCTION.md. Results are partial reproduction and
   require a full MLHAT implementation for an exact-paper replication claim.
