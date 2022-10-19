trec_eval -a qrels.txt %1-results.txt > %1-metrics.txt
trec_eval -q qrels.txt %1-results.txt >> %1-metrics.txt