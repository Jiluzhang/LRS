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


zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg19.txt  # 5658
zcat GSM2970932_sciATAC_GM12878_counts.txt.gz | grep GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT | sed 's/_/\t/g' | cut -f 1,2,3,5 | sort -k1,1 -k2,2n > GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg19.txt  # 6904

liftOver TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg19.txt hg19ToHg38.over.chain.gz TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38.txt unmapped.bed && rm unmapped.bed  # 5656
liftOver GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg19.txt hg19ToHg38.over.chain.gz GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38.txt unmapped.bed && rm unmapped.bed  # 6904

java -jar juicer_tools_1.19.02.jar dump oe KR ~/scHiC/bulk_hic/loop_subtype/GSE118911_Fulco-2018-mES_combined_30.hic \
                                              10:69340000:69340000 10:69420000:69420000 BP 10000 | sed '1d' | cut -f 3

bedtools makewindows -g hg38.chrom.sizes -w 1000 > hg38_1kb_bins.bed

bedtools intersect -a hg38_1kb_bins.bed -b TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38.txt -wa | uniq > TAATGCGCTTGATTGGCGTCAAGATAGTGGCTCTGA_hg38_1kb.bed  # 9592
bedtools intersect -a hg38_1kb_bins.bed -b GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38.txt -wa | uniq > GAGATTCCGAACTCGACTTTAATTAGCCCAGGACGT_hg38_1kb.bed  # 11582












