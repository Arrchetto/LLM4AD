# Dataset Source and Validation Record

## Problem family

Online 1D Bin Packing

## LLM4AD tasks

- online_bin_packing
- online_bin_packing_2O

## Dataset

FunSearch Weibull protocol

## Source page

https://github.com/google-deepmind/funsearch

## Generator

LLM4AD/llm4ad/task/optimization/online_bin_packing/generate_weibull_instances.py

## Protocol

Weibull(scale=45, shape=3), clipped to [1, capacity], rounded to nearest integer, seed 2024

## Download date

2026-07-01

## Maintainer

Google DeepMind FunSearch team

## Citation

Romera-Paredes et al. (2023). Mathematical discoveries from program search with large language models. Nature. DOI:10.1038/s41586-023-06924-6

## License/terms

FunSearch software is Apache-2.0; other materials are CC BY 4.0. Cite Romera-Paredes et al., Nature, DOI:10.1038/s41586-023-06924-6.

## Raw files

- {'name': 'generated/test_10000.json', 'size': 499976, 'sha256': '0f91def5fdf4e2cfc5cd026b2d15558689ecb404120ba43468ab4aec333ee18f'}
- {'name': 'generated/test_100000.json', 'size': 999196, 'sha256': 'd5001ab7baad1a1bb799930f0d46301c4233cf19a9578e1ecf1f0714ec33bdfa'}
- {'name': 'generated/test_5000.json', 'size': 250215, 'sha256': '683cd8559fc3721e4af7850ebb1a3bb283ddd4c371247f78707d5658d32ca809'}
- {'name': 'generated/train_5000.json', 'size': 250215, 'sha256': 'f5c972b14e91f857bdaf67b73a4f432af880fad554c91831af7f98c7d7befebd'}
- {'name': 'generated/val_5000.json', 'size': 250193, 'sha256': '7ce6669376f1039b917b59d892499b62d1ca5123e07a630954259f8022ffc006'}

## Instance count

21
