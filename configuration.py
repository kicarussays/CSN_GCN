"""
Written on October 29, 2020

This provides parameters.

"""

import argparse

clinical_filepath = 'data/Subtype_Info.csv'
gene_data = 'data/GEM_fs_0001.csv'
train_path = 'data/GEM_train_fs.csv'
test_path = 'data/GEM_test_fs.csv'
mode = 'standard'
gcn_mode = True


args = argparse.ArgumentParser(description='CSN project using GCN')

args.add_argument('--batch', type=int, default=64)
args.add_argument('--filter', type=int, default=8)

args = args.parse_args()







