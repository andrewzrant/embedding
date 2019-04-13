import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community.centrality import girvan_newman

def addPairs(train_transac,connectCol):
    device_codeGraph = train_transac[["UID",connectCol]].drop_duplicates()
    device_codeGraph = device_codeGraph[-device_codeGraph[connectCol].isnull()]
    device_toID = device_codeGraph.copy()
    device_toID.columns = ['ToUID', connectCol]
    IDPairs = pd.merge(device_codeGraph,device_toID )
    IDPairs = IDPairs[IDPairs.UID != IDPairs.ToUID]
    IDPairs = IDPairs[["UID","ToUID" ]]
    IDPairs = IDPairs.drop_duplicates()
    return IDPairs

for_dev_graph = pd.read_csv("dev_graph_data.csv",encoding="utf8")
use_col = ["UID","device_code12Int","device_code22Int","device_code32Int","mac12Int","mac22Int"]
for_dev_graph = for_dev_graph[use_col]
IDP1 = addPairs(for_dev_graph,"device_code12Int")
IDP2 = addPairs(for_dev_graph,"device_code22Int")
IDP3 = addPairs(for_dev_graph,"device_code32Int")
IDP4 = addPairs(for_dev_graph,"mac12Int")
IDP5 = addPairs(for_dev_graph,"mac22Int")


IDP = pd.concat([ IDP1,IDP2,IDP3,IDP4,IDP5])
# IDP = pd.concat([ IDP1,IDP2])
IDP = IDP.drop_duplicates()
# # 返回df 最后concat起来去重 然后再加边chenglist
IDPairs = IDP.as_matrix().tolist()

G = nx.Graph()
G.add_edges_from(IDPairs)
co = nx.connected_components(G)
cos = sorted(co, key = len, reverse=True)
commNames = []
commMembs = []
for n,m in enumerate(cos):
    cnls = [n] * len(m)
    commNames = commNames + cnls
    commMembs = commMembs +list( m)
nxcommdfs = pd.DataFrame()
nxcommdfs["connected_components"] = commNames
nxcommdfs["UID"] = commMembs

nxcommdfs.to_csv("dev_connected_components.csv",encoding="utf8",index=False)