## workdir: /fs/home/jiluzhang/LRS/GNN

conda create -n GNN python=3.9
conda activate GNN

conda install conda-forge::pytorch_geometric
conda install pytorch::pytorch
conda install ipython

######## self-supervised learning ########
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, GAE

## data preparation
node_features = torch.tensor([[0.8, 0.2, 0.5],
                              [0.6, 0.3, 0.1],
                              [0.7, 0.5, 0.0],
                              [0.2, 0.9, 0.4],
                              [0.4, 0.7, 0.8]], dtype=torch.float)

edge_index = torch.tensor([[0, 1, 2, 3],
                           [1, 2, 3, 4]], dtype=torch.long)  # source node -> target node

edge_weights = torch.tensor([0.9, 0.7, 0.3, 0.5], dtype=torch.float)

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

## construct model
class GAE_Imputer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # self.encoder = torch.nn.Sequential(GCNConv(in_dim, hidden_dim),
        #                                    torch.nn.ReLU(),
        #                                    GCNConv(hidden_dim, hidden_dim))  # torch.nn.Sequential only support a single input
        
        self.encoder = Sequential('x, edge_index', [(GCNConv(in_dim, hidden_dim), 'x, edge_index -> x'), 
                                                    torch.nn.ReLU(inplace=True), 
                                                    (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x')])
        
        # 解码器：内积重构边权重
        self.decoder = lambda z, edge_index: torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)  # 节点编码
        edge_pred = self.decoder(z, edge_index)  # 预测边权重
        return edge_pred

model = GAE_Imputer(in_dim=3, hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

## train model
def train():
    model.train()
    optimizer.zero_grad()
    # 仅用观测到的边训练
    pred_weights = model(data.x, data.edge_index)
    loss = F.mse_loss(pred_weights, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

## predict all edges
all_possible_edges = torch.combinations(torch.arange(5), 2).t()

model.eval()
with torch.no_grad():
    z = model.encoder(data.x, data.edge_index)
    imputed_weights = model.decoder(z, all_possible_edges)

for i in range(all_possible_edges.size(1)):
    src, dst = all_possible_edges[:, i]
    print(f'Edge {src}-{dst}: Predicted Weight = {imputed_weights[i]:.4f}')

# Edge 0-1: Predicted Weight = 0.9239
# Edge 0-2: Predicted Weight = 0.6844
# Edge 0-3: Predicted Weight = 0.3434
# Edge 0-4: Predicted Weight = 0.3373
# Edge 1-2: Predicted Weight = 0.7064
# Edge 1-3: Predicted Weight = 0.3239
# Edge 1-4: Predicted Weight = 0.3179
# Edge 2-3: Predicted Weight = 0.4432
# Edge 2-4: Predicted Weight = 0.4350
# Edge 3-4: Predicted Weight = 0.5492



######## supervised learning ########
# conda install conda-forge::scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from torch_geometric.utils import negative_sampling

import pandas as pd
from tqdm import *

def generate_random_graph(num_nodes, num_edges, feat_dim=16):
    # 生成节点特征和边
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 生成边标签（正样本=1，负样本=0）
    edge_label = torch.ones(edge_index.size(1))  # 正样本
    neg_edges = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_edges)  # 负采样
    full_edge_index = torch.cat([edge_index, neg_edges], dim=1)
    full_edge_label = torch.cat([edge_label, torch.zeros(neg_edges.size(1))])
    
    return Data(x=x, edge_index=edge_index, edge_label=full_edge_label, edge_label_index=full_edge_index)

# 生成5个随机图作为数据集
dataset = [generate_random_graph(num_nodes=np.random.randint(200, 500),  # 每个图20~50个节点
                                 num_edges=np.random.randint(3000, 10000)) for _ in range(10)]

# 划分训练集和测试集
train_dataset = dataset[:8]
test_dataset = dataset[8:]

class MultiGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = TransformerConv(hidden_dim*heads, hidden_dim, heads=1, dropout=0.2)
        self.decoder = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
    
    def forward(self, x, edge_index, batch):
        # 节点编码
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)  # [num_nodes_total, hidden_dim]
        
        return h
    
    def decode(self, h, edge_index):
        # 计算节点对(u,v)的特征拼接
        h_src = h[edge_index[0]]  # 源节点特征
        h_dst = h[edge_index[1]]  # 目标节点特征
        edge_feats = torch.cat([h_src, h_dst], dim=-1)
        return torch.sigmoid(self.decoder(edge_feats)).squeeze()  # 输出概率
    
    def loss(self, pred, label):
        return F.binary_cross_entropy(pred, label)

# 初始化模型和优化器
model = MultiGraphTransformer(in_dim=16, hidden_dim=32, heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index, batch.batch)
        pred = model.decode(h, batch.edge_label_index)
        loss = model.loss(pred, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            h = model(batch.x, batch.edge_index, batch.batch)
            pred = model.decode(h, batch.edge_label_index)
            preds.append(pred.cpu())
            labels.append(batch.edge_label.cpu())
    
    pred_all = torch.cat(preds).numpy()
    label_all = torch.cat(labels).numpy()
    auc = roc_auc_score(label_all, pred_all)
    ap = average_precision_score(label_all, pred_all)
    return auc, ap

# 训练循环
for epoch in range(1, 101):
    loss = train()
    train_auc, train_ap = test(train_loader)
    test_auc, test_ap = test(test_loader)
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, '
              f'Train AP: {train_ap:.4f}, Test AP: {test_ap:.4f}')


## GM127878 scATAC-seq data (hg19)
wget -c https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2970nnn/GSM2970932/suppl/GSM2970932_sciATAC_GM12878_counts.txt.gz

## GM12878 Hi-C data (GRCh38)
wget -c https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/5809b32e-0aea-4cf5-a174-cf162d591a35/4DNFI9YAVTI1.hic

zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | cut -f 2 | sort | uniq > GSM2970932_sciATAC_GM12878_cells.txt

zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg19.txt  # 10854
zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg19.txt  # 655
zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg19.txt  # 5658
zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg19.txt  # 6904

liftOver CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg19.txt hg19ToHg38.over.chain.gz CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg38.txt unmapped.bed && rm unmapped.bed  # 10851
liftOver ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg19.txt hg19ToHg38.over.chain.gz ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg38.txt unmapped.bed && rm unmapped.bed  # 655
liftOver TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg19.txt hg19ToHg38.over.chain.gz TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38.txt unmapped.bed && rm unmapped.bed  # 5656
liftOver GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg19.txt hg19ToHg38.over.chain.gz GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38.txt unmapped.bed && rm unmapped.bed  # 6904

java -jar juicer_tools_1.19.02.jar dump oe KR 4DNFI9YAVTI1.hic 1 1 BP 1000 | sed '1d' | awk '{if($2-$1<=1000000 && $2-$1>0) print $0}' | awk '{if($3>2 && $3<5) print $0}' | wc -l

bedtools makewindows -g hg38.chrom.sizes -w 1000 > hg38_1kb_bins.bed

bedtools intersect -a hg38_1kb_bins.bed -b CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg38.txt -wa | uniq > CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg38_1kb.bed  # 18218
bedtools intersect -a hg38_1kb_bins.bed -b ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg38.txt -wa | uniq > ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg38_1kb.bed  # 1136
bedtools intersect -a hg38_1kb_bins.bed -b TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38.txt -wa | uniq > TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38_1kb.bed  # 9592
bedtools intersect -a hg38_1kb_bins.bed -b GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38.txt -wa | uniq > GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38_1kb.bed  # 11582

cat CGCTCATTTTGATACGATTCAAGATAGTGGCTCTGA_hg38_1kb.bed ATTCAGAAGCATATGAGCCAGCCGGCTTGTACTGAC_hg38_1kb.bed \
    TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38_1kb.bed GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38_1kb.bed |\
    sort -k1,1 -k2,2n | uniq > hg38_1kb_peaks.bed

java -jar juicer_tools_1.19.02.jar dump oe KR 4DNFI9YAVTI1.hic 1 1 BP 1000 | sed '1d' | awk '{if($2-$1<=1000000 && $2-$1>0) print $0}' |\
                                                                             awk '{if($3>2 && $3<5) print $0}' > hic_chr1.txt  # 2262219

for d in `seq 10000000 100000 19999999`; do
    awk '{if($1=="chr1") print $0}' hg38_1kb_bins.bed | awk '{if($2>='"$d"' && $2<('"$d"'+100000)) print $0}' | awk '{print $0 "\t" NR-1}' > hg38_1kb_bins_chr1_100kb_test.bed
    awk '{if($1>='"$d"' && $1<('"$d"'+100000) && $2>='"$d"' && $2<('"$d"'+100000)) print $0}' hic_chr1.txt > hg38_1kb_bins_chr1_100kb_test_hic.txt
    awk '{print "chr1" "\t" $1 "\t" $1+1000}' hg38_1kb_bins_chr1_100kb_test_hic.txt | bedtools intersect -a stdin -b hg38_1kb_bins_chr1_100kb_test.bed -wa -wb | awk '{print $7}' > left.txt
    awk '{print "chr1" "\t" $2 "\t" $2+1000}' hg38_1kb_bins_chr1_100kb_test_hic.txt | bedtools intersect -a stdin -b hg38_1kb_bins_chr1_100kb_test.bed -wa -wb | awk '{print $7}' > right.txt
    paste left.txt right.txt > left_right_$d.txt
    rm hg38_1kb_bins_chr1_100kb_test.bed hg38_1kb_bins_chr1_100kb_test_hic.txt left.txt right.txt
    echo $d done
done



num_nodes=1000
feat_dim=16
num_edges=(99+1)*99/2=4950



m = np.zeros([2, 4950], dtype=int)
t = 0
for i in range(100):
    for j in range(100):
        if j>i:
            m[0][t] = i
            m[1][t] = j
            t += 1

set_0_1 = set(tuple(col) for col in m.T)

dataset = []
for d in tqdm(range(10000000, 20000000, 100000), ncols=80, desc='generate graphs'):
    x = torch.randn(100, 16)
    ## add open or close info
    idx_0_1 = torch.arange(0, 100)
    idx_1 = torch.randperm(100)[:10]
    idx_0 = idx_0_1[~torch.isin(idx_0_1, idx_1)]
    x[idx_0, 0] += 1
    x[idx_0, 1] += 1
    
    edge_index_df = pd.read_table('left_right_'+str(d)+'.txt', header=None)
    edge_index = torch.tensor(np.array(([edge_index_df[0].values, edge_index_df[1].values])))
    edge_label = torch.ones(edge_index.size(1))
    
    #neg_edges = negative_sampling(edge_index, num_nodes=1000, num_neg_samples=10*edge_index.size(1), force_undirected=True)
    set_1 = set(tuple(col) for col in np.array(([edge_index_df[0].values, edge_index_df[1].values])).T)
    neg_edges = torch.tensor(np.array(list(set_0_1 - set_1)).T)
    
    full_edge_index = torch.cat([edge_index, neg_edges], dim=1)
    full_edge_label = torch.cat([edge_label, torch.zeros(neg_edges.size(1))])
    dataset.append(Data(x=x, edge_index=edge_index, edge_label=full_edge_label, edge_label_index=full_edge_index))


# 划分训练集和测试集
train_dataset = dataset[:70]
test_dataset = dataset[70:]

class MultiGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = TransformerConv(hidden_dim*heads, hidden_dim, heads=1, dropout=0.2)
        self.decoder = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
    
    def forward(self, x, edge_index, batch):
        # 节点编码
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)  # [num_nodes_total, hidden_dim]
        return h
    
    def decode(self, h, edge_index):
        # 计算节点对(u,v)的特征拼接
        h_src = h[edge_index[0]]  # 源节点特征
        h_dst = h[edge_index[1]]  # 目标节点特征
        edge_feats = torch.cat([h_src, h_dst], dim=-1)
        return torch.sigmoid(self.decoder(edge_feats)).squeeze()  # 输出概率
    
    def loss(self, pred, label):
        return F.binary_cross_entropy(pred, label)

# 初始化模型和优化器
model = MultiGraphTransformer(in_dim=16, hidden_dim=32, heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index, batch.batch)
        pred = model.decode(h, batch.edge_label_index)
        loss = model.loss(pred, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            h = model(batch.x, batch.edge_index, batch.batch)
            pred = model.decode(h, batch.edge_label_index)
            preds.append(pred.cpu())
            labels.append(batch.edge_label.cpu())
    
    pred_all = torch.cat(preds).numpy()
    label_all = torch.cat(labels).numpy()
    auc = roc_auc_score(label_all, pred_all)
    ap = average_precision_score(label_all, pred_all)
    return auc, ap

# model.eval()
# preds, labels = [], []
# i = 0
# with torch.no_grad():
#     for batch in test_loader:
#         h = model(batch.x, batch.edge_index, batch.batch)
#         print(i, h)
#         pred = model.decode(h, batch.edge_label_index)
#         preds.append(pred.cpu())
#         labels.append(batch.edge_label.cpu())
#         if i==1:
#             break
#         i += 1


# 训练循环
for epoch in range(1, 201):
    loss = train()
    train_auc, train_ap = test(train_loader)
    test_auc, test_ap = test(test_loader)
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, '
              f'Train AP: {train_ap:.4f}, Test AP: {test_ap:.4f}')

