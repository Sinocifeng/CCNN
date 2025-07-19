import math
import pickle
import random
import threading
from collections import Counter

import numpy as np

# Generate a network(nxn), edge probability p between nodes


def generate_adj_matrix(n, p = 0.5):
    """生成 n×n 无向图的 0/1 邻接矩阵（含对角线自环）"""
    upper = np.triu(np.random.rand(n, n) < p, k=1)
    adj = (upper + upper.T).astype(int)
    np.fill_diagonal(adj, 1)
    return adj.tolist()
    
# Generate random attribute information for the network matrix    
def _split_numbers(n: int) -> list[int]:
    step = 1024 / (n + 1)
    return [int((i + 1) * step) - 1 for i in range(n)]

def generate_attr(adj, attr_num):
    """为每个节点附加 10 位二进制属性，返回 (属性邻接矩阵, 状态列表)"""
    attrs = _split_numbers(attr_num)
    m = []
    for row in adj:
        vec = list(map(int, f'{random.choice(attrs):010b}'))
        m.append(row + vec)
    return m, attrs

# Calculate all equiconcepts in social network
def equiconcepts_cal(adj):
    adj = np.array(adj, dtype=bool)
    n = adj.shape[0]

    intents = {tuple(np.where(row)[0] + 1) for row in adj}

    for cur in list(intents):
        for other in list(intents):
            inter = tuple(sorted(set(cur) & set(other)))
            if inter:
                intents.add(inter)

    equiconcepts = []
    for intent in intents:
        mask = np.ones(n, dtype=bool)
        for j in intent:
            mask &= adj[:, j - 1]
        extent = tuple(np.where(mask)[0] + 1)
        if extent == intent:
            equiconcepts.append(list(intent))
    return equiconcepts

# get all fair equiconcepts in the equaiconcepts
def fair_equiconcepts_cal(attr_adj, equi_lst, states):
    n, m0 = len(attr_adj), len(attr_adj[0])
    attr_start = n
    res = []
    for nodes in equi_lst:
        cnt = Counter(
            int(''.join(map(str, attr_adj[v - 1][attr_start:])), 2)
            for v in nodes
        )
        vals = [cnt.get(s, 0) for s in states]
        if max(vals) - min(vals) <= 1:
            res.append(nodes)
    return res



def convert_vector(equiconcepts_lst, node_num):
    mat = np.zeros((len(equiconcepts_lst), node_num))
    for i, ids in enumerate(equiconcepts_lst):
        mat[i, np.array(ids) - 1] = 1
    return mat.astype(int)

# Generate .pkl files
def gen_dataset(nodefile_num, attr_num, edge_p=0.5):
    idx = index_map[nodefile_num]
    for _ in range(200):
        n = random.randint(nodefile_num - 5, nodefile_num)
        adj = generate_adj_matrix(n, edge_p)
        attr_adj, states = generate_attr(adj, attr_num)

        equi = equiconcepts_cal(adj)
        fair = fair_equiconcepts_cal(attr_adj, equi, states)

        data = {
            'attr_context': np.array(attr_adj, dtype=int),
            'equiconcepts': convert_vector(equi, n),
            'fair_equiconcepts': convert_vector(fair, n)
        }

        path = f"../data/{nodefile_num}x{nodefile_num}/{idx:05d}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        idx += 1
    index_map[nodefile_num] = idx            # 更新全局计数


index_map = {15: 0, 25: 0, 35: 0, 45: 0, 55: 0} # Global Counter
if __name__ == '__main__':
    edge_p = 0.1
    tasks = [(15, 3), (25, 4), (35, 3), (45, 4), (55, 5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        for n, a in tasks:
            pool.submit(gen_dataset, n, a, edge_p)