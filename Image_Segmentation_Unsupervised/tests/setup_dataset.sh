#!/bin/bash
# mkdir -p data/test
# mkdir -p data/test/0
# mkdir -p data/test/1
wget --tries=3 --timeout=30 -P data https://github.com/class-neural-network-2025/data-07/raw/refs/heads/main/test.tar.gz;
tar -xzf data/test.tar.gz -C data/;