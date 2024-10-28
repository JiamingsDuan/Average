# -*- coding: utf-8 -*-
import pandas as pd

data721 = pd.read_csv('data/big_five_721.csv')[['G', 'No', 'N', 'E', 'O', 'A', 'C']]
data664 = pd.read_csv('data/feature_25.csv')
data664.insert(1, 'G', value=None)
data664.insert(2, 'N', value=None)
data664.insert(3, 'E', value=None)
data664.insert(4, 'O', value=None)
data664.insert(5, 'A', value=None)
data664.insert(6, 'C', value=None)
data664.insert(7, 'T', value=None)

for i in list(data664['No']):

    if i in list(data721['No']):
        index_721 = data721[data721['No'] == i].index[0]
        value_n = data721.iloc[index_721]['N']
        value_e = data721.iloc[index_721]['E']
        value_o = data721.iloc[index_721]['O']
        value_a = data721.iloc[index_721]['A']
        value_c = data721.iloc[index_721]['C']
        value_g = data721.iloc[index_721]['G']
        index_664 = data664[data664['No'] == i].index[0]
        data664.loc[index_664, 'N'] = value_n
        data664.loc[index_664, 'E'] = value_e
        data664.loc[index_664, 'O'] = value_o
        data664.loc[index_664, 'A'] = value_a
        data664.loc[index_664, 'C'] = value_c
        trait_set = [value_n, value_e, value_o, value_a, value_c]
        trait = trait_set.index(max(trait_set)) + 1
        print(trait)
        data664.loc[index_664, 'T'] = trait
        if value_g == '男':
            value_g = 0
        else:
            value_g = 1
        data664.loc[index_664, 'G'] = value_g
    else:

        pass
data664.to_csv('data/data_664.csv', sep=',', index=False)
