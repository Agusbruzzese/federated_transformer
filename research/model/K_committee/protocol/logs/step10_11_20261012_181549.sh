set -e
cd /home/augustin/Documents/federated_transformer
P=research/model/K_committee/protocol
.venv/bin/python $P/verify_freeze.py
.venv/bin/python $P/launch.py run ablation 4
.venv/bin/python $P/verify_freeze.py
.venv/bin/python $P/score_all.py main 12
SM=$(ls -t $P/scoring/scores_main_*.parquet | head -1); MM=$(ls -t $P/scoring/models_main_*.parquet | head -1)
SD=$P/scoring/scores_dryrun_20261011_223009.parquet; MD=$P/scoring/models_dryrun_20261011_223009.parquet
STAMP=$(date +%Y%m%d_%H%M%S)
.venv/bin/python -c "import pandas as pd,sys,os; a=sys.argv; o1,o2=a[5],a[6]; assert not os.path.exists(o1) and not os.path.exists(o2); pd.concat([pd.read_parquet(a[1]),pd.read_parquet(a[2])],ignore_index=True).to_parquet(o1); pd.concat([pd.read_parquet(a[3]),pd.read_parquet(a[4])],ignore_index=True).to_parquet(o2); print('combined',o1,o2)" $SD $SM $MD $MM $P/scoring/scores_combined_$STAMP.parquet $P/scoring/models_combined_$STAMP.parquet
.venv/bin/python $P/analyze.py main $P/scoring/scores_combined_$STAMP.parquet $P/scoring/models_combined_$STAMP.parquet $P/results_controls_20261011_231043.json
echo CHAIN_EXIT 0
