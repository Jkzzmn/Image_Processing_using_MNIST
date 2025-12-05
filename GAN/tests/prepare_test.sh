#!/bin/bash
echo "================================================="
echo "install dependencies and prepare data for testing"
echo "================================================="
apt-get install -y wget
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade pip

echo "================================================="
echo "clean previous test results and prepare for sampling"
echo "================================================="
rm -f result_autograding.csv
rm -rf data/sample
mkdir -p data/sample

# echo "================================================="
# echo "download and prepare dataset for testing"
# echo "================================================="
# rm -rf data/test
# mkdir -p data/test
# wget --tries=3 --timeout=30 -P data https://github.com/class-neural-network-2025/data-OO/raw/refs/heads/main/test.tar.gz;
# tar -xzf data/test.tar.gz -C data/;
