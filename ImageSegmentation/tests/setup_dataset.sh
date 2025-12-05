#!/bin/bash
wget --tries=3 --timeout=30 -P data https://github.com/class-neural-network-2025/data-06/raw/refs/heads/main/test.tar.gz;
tar -xzf data/test.tar.gz -C data/;