#!/bin/bash
PATH=/usr/bin/python3:$PATH
for seed in {1..5}; do
  echo "Seed: $seed"
  python3 removing_years.py $seed bert-base-multilingual-cased
  python3 removing_years.py $seed camembert-base
  python3 removing_years.py $seed xlm-roberta-large --fp16
done
