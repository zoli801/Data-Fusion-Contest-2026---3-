import pandas as pd
import numpy as np

d_h = pd.read_csv('../data_heroes.csv')
d_o = pd.read_csv('../data_objects.csv')
d_s = pd.read_csv('../dist_start.csv')
d_d = pd.read_csv('../dist_objects.csv')

r = 500
vc = 100
hc = 2500
n = len(d_o)

m = d_d.values
np.fill_diagonal(m, 999999)

md = np.minimum(d_s['dist_start'].values, m.min(axis=0))
req = np.sum(md) + n * vc

def f(h):
    mp = d_h['move_points'].iloc[:h].sum() * 7
    sav = h * 7 * 99
    return mp + sav >= req

h_min = next((h for h in range(1, 101) if f(h)), 100)
mx = n * r - h_min * hc

print(mx)
