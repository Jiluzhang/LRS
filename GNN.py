## workdir: /fs/home/jiluzhang/LRS/GNN

conda create -n GNN python=3.9
conda activate GNN

conda install conda-forge::pytorch_geometric
conda install pytorch::pytorch
conda install ipython


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





