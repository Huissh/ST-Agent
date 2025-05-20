import dgl
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
import dgl.function as fn
import networkx as nx
from datetime import datetime
import pickle
import torch.nn.functional as F
import random

class ResearchGraphDataset:
    def __init__(self,df,splits=None,max_authors = 2):
        self.df = df.sort_values('publication_year')
        self.max_authors = max_authors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = SentenceTransformer('model/all-MiniLM-L6-v2').to(self.device)

        # 预生成全局数据
        self.author_metadata = self._preprocess_authors()
        self.all_edges = self._preprocess_collaborations()
        self._init_first_collab_year() 
        
        self._init_author_metadata()
        if splits == None:
            self.splits = self.get_time_splits()
        else:
            self.splits = splits

    def _preprocess_authors(self):
        '''预处理作者元数据'''
        author_data = defaultdict(lambda: {
            'papers': [],
            'total_citations': 0,
            'collaborators': defaultdict(int),
            'paper_count': 0,
            'avg_collab': 0
        })
        
        # 遍历所有论文
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing papers"):
            # 提取本论文所有作者ID
            authors = []
            for i in range(1, self.max_authors+1): #添加一二作者
                if pd.notna(row[f'author{i}_id']):
                    authors.append(row[f'author{i}_id'])
            if pd.notna(row[f'corresponding_author_ids']): #添加通讯作者
                authors.append(row[f'corresponding_author_ids'])
            
            # 更新作者信息
            for author_id in authors:
                # 记录论文信息
                author_data[author_id]['papers'].append({
                    'doi': row['doi'],
                    'year': row['publication_year'],
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'keywords': row['keyword'],
                    'question': row['abstract_question'],
                    'method': row['abstract_method'],
                    'citations': row['cited_by_count'],
                    'author_count': row['author_num'],
                })
                # 累计被引量
                author_data[author_id]['total_citations'] += row['cited_by_count']
                
            # 记录合作关系（无向图）
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a1, a2 = authors[i], authors[j]
                    author_data[a1]['collaborators'][a2] += 1
                    author_data[a2]['collaborators'][a1] += 1
        
        return dict(author_data)

    def _preprocess_collaborations(self):
        '''预处理合作关系，保留每次独立合作事件'''
        edges = []
        # 遍历所有论文
        for _, row in tqdm(self.df.iterrows(), desc="Processing collaborations"):
            # 提取本论文的年份和作者列表
            authors = []
            for i in range(1, self.max_authors+1):
                author_id = row.get(f'author{i}_id')
                if pd.notna(author_id):
                    authors.append(author_id)
            if pd.notna(row[f'corresponding_author_ids']): #添加通讯作者
                authors.append(row[f'corresponding_author_ids'])
            
            # 生成所有无向边组合（每篇论文独立记录）
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a, b = sorted([authors[i], authors[j]])  # 保持无向性
                    edges.append({
                        'source': a,
                        'target': b,
                        'year': row['publication_year'],
                        'doi': row['doi']
                    })
        
        edge_df = pd.DataFrame(edges)
        if not edge_df.empty:
            # 按年度统计合作次数
            edge_df = edge_df.groupby(['source', 'target', 'year']).agg(
                papers_num=('doi', 'count'),
                papers=('doi', lambda x: list(x))
            ).reset_index()
        
        return edge_df
        
    def _init_first_collab_year(self):
        '''记录每个合作对的首次出现年份'''
        self.first_collab_year = {}
        for year, group in self.all_edges.groupby('year'):
            for _, row in group.iterrows():
                pair = tuple(sorted((row['source'], row['target'])))
                if pair not in self.first_collab_year:
                    self.first_collab_year[pair] = year
                    
    def _init_author_metadata(self):
        self.author_meta = {}
        """获取原始文本"""
        for aid, meta in self.author_metadata.items():
            # 合并原始文本数据
            raw_texts = []
            for p in meta['papers']:
                raw_text = f"{p['title']} {p['abstract']} {p['keywords']} {p['question']} {p['method']}"
                raw_texts.append(raw_text)
            
            # 存储完整元数据
            self.author_meta[aid] = {
                'raw_text': ' '.join(raw_texts[:5]),  # 保留最近5篇的完整文本
                'raw_info': [{
                    'year': p['year'],
                    'title': p['title'],
                    'citations': p['citations']
                } for p in meta['papers']],
                'raw_citations': meta['total_citations'],
                'raw_paper_count': len(meta['papers'])}

    def get_time_splits(self):
        '''生成严格时序划分'''
        years = sorted(self.df['publication_year'].unique())
        splits = {} 
        
        answer = input("你是否要重新运行 build_historical_graph？还是补充参数直接用本地Graph（是/否）：")
        if answer.strip() == "是":
            # 替换成实际的年份
            pass
        else:
            print("操作已中止。")
            return None

        # 确保有足够的后续年份
        for t_idx in range(len(years)):
            train_end = int(years[t_idx])
            
            # 训练图包含截止train_end的数据
            train_graph = self.build_historical_graph(train_end) #构建的是2023的图但是下面提取数据是小于2023
            
            splits[train_end]={
                'train': train_graph
            }

        #获取当前时间戳，以帮助命名和qu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"splits_{timestamp}.pkl"
        
        # 统一保存为 pickle 文件
        with open(filename, "wb") as f:
            pickle.dump(splits, f)

        return splits

    def generate_authorid2idx(self, end_year): # id转来转去
        # 筛选历史边
        history_edges = self.all_edges[self.all_edges['year'] < end_year]
        merged_edges = history_edges.groupby(['source', 'target']).agg(papers_num=('papers_num', 'sum'),year=('year', list),
                                                                       all_papers=('papers', lambda x: sum(x, []))).reset_index()
        # 获取活跃作者
        active_authors = pd.unique(merged_edges[['source', 'target']].values.ravel('K'))#按照缓存顺序储存 信息越多则缓存越多 越多先存储
        #author_idx = {aid: i for i, aid in enumerate(active_authors)}
        author_id_to_idx = {aid: idx for idx, aid in enumerate(active_authors)}
        idx_to_author_id = {idx: aid for idx, aid in enumerate(active_authors)}
        return active_authors,author_id_to_idx,idx_to_author_id
        
    def build_historical_graph(self, end_year: int): #选出截止到end_year-1的数据
        history_edges = self.all_edges[self.all_edges['year'] < end_year]
        merged_edges = history_edges.groupby(['source', 'target']).agg(papers_num=('papers_num', 'sum'),year=('year', list),all_papers=('papers', lambda x: sum(x, []))).reset_index()

        active_authors,author_id_to_idx,idx_to_author_id = self.generate_authorid2idx(end_year) #获取的是23的id

        # 创建DGL图
        num_nodes = int(len(active_authors))
        g = dgl.graph(([], []), num_nodes=num_nodes)
        # 节点特征
        features = defaultdict(list)
        for aid in active_authors:
            # 研究主题嵌入（最近5篇论文） 
            history_papers = [p for p in self.author_metadata[aid]['papers'] if p['year'] < end_year]
            recent_papers = sorted(history_papers, key=lambda x: x['year'], reverse=True)[:5] 
            text = ' '.join([f"{p['title']} {p['abstract']} {p['keywords']} {p['question']} {p['method']}" for p in recent_papers])
            embedding = self.text_encoder.encode(text, convert_to_tensor=True)
            features['text_emb'].append(embedding.to(dtype=torch.float32))

            features['citations'].append(self.author_metadata[aid]['total_citations'])

            # 数值特征
            features['avg_citations'].append(sum(p['citations'] for p in history_papers) / len(history_papers) if history_papers else 0) 
            features['paper_count'].append(len(history_papers))
            features['avg_collab'].append(np.mean([p['author_count'] for p in history_papers]) if history_papers else 0)


        g.ndata['text_emb'] = torch.stack(features['text_emb']).cpu()

        g.ndata['citations'] = torch.tensor(features['citations'], dtype=torch.float32)
        g.ndata['avg_citations'] = torch.tensor(features['avg_citations'], dtype=torch.float32)
        g.ndata['paper_count'] = torch.tensor(features['paper_count'], dtype=torch.float32)
        g.ndata['avg_collab'] = torch.tensor(features['avg_collab'], dtype=torch.float32)


        src_ids = [int(author_id_to_idx[row['source']]) for _, row in merged_edges.iterrows()]
        dst_ids = [int(author_id_to_idx[row['target']]) for _, row in merged_edges.iterrows()]
        src_tensor = torch.tensor(src_ids, dtype=torch.int64)
        dst_tensor = torch.tensor(dst_ids, dtype=torch.int64)

        g.add_edges(src_tensor, dst_tensor)


        years_tensors = [torch.tensor(y, dtype=torch.int16) for y in merged_edges['year']]
        papers_tensors = [torch.tensor([hash(p) for p in ps], dtype=torch.long) for ps in merged_edges['all_papers']]

        #print(f"papers_tensors: {len(merged_edges['papers_num'])}")
        g.edata['papers_num'] = torch.tensor(merged_edges['papers_num'].values, dtype=torch.float32)
        g.edata['year'] = pad_sequence(years_tensors, batch_first=True, padding_value=0)
        g.edata['papers'] = pad_sequence(papers_tensors, batch_first=True, padding_value=-1)
        g = g.to(self.device)

        return g
    
    def _add_topological_features(self, g): #后来没用上
        '''添加拓扑特征'''  
        # 点度中心性
        end_year = g.edata['year'].max().item()
        active_authors,author_id_to_idx,idx_to_author_id = self.generate_authorid2idx(end_year)
        active_author_indices = torch.tensor([author_id_to_idx[aid] for aid in active_authors], dtype=torch.long)
        
        nx_g = g.cpu().to_networkx().to_undirected()
        deg_cent = nx.degree_centrality(nx_g)
        constraint_nx = nx.constraint(nx_g)
        g.ndata['degree_cent'] = torch.zeros(g.num_nodes(), device=g.device)
        g.ndata['constraint'] = torch.ones(g.num_nodes(), device=g.device)  
        for nid in active_author_indices:
            g.ndata['degree_cent'][nid.item()] = deg_cent[nid.item()]
            g.ndata['constraint'][nid.item()] = constraint_nx[nid.item()]
        return g

    def _get_author_embedding(self, author_id: str, end_year: int) -> np.ndarray:
        '''获取作者截至某年的文本嵌入'''
        papers = [p for p in self.author_metadata[author_id]['papers'] if p['year'] < end_year]
        recent_texts = [' '.join(f"{p['title']}{p['abstract']}{p['keywords']}") for p in papers[:5]] #最近5篇论文 
        if not recent_texts:
            return torch.zeros(384, dtype=torch.float32, device=self.device)
        embedding = self.text_encoder.encode(recent_texts, convert_to_tensor=True, device=self.device, output_value='sentence_embedding')
        return embedding 

    def attention_pooling(self,paper_embs: torch.Tensor) -> torch.Tensor:
        if paper_embs.ndim == 1:
            return paper_embs.view(1, -1)
    
        query = paper_embs.mean(dim=0, keepdim=True)  # [1, D]
        scores = torch.matmul(query, paper_embs.T) / (paper_embs.shape[1] ** 0.5)  # [1, N]
        attn_weights = F.softmax(scores, dim=-1)  # [1, N]
        author_emb = torch.matmul(attn_weights, paper_embs)  # [1, D]

        return author_emb

    def _is_historical_new_pair(self, pair: tuple, target_year: int) -> bool:
        '''是否为历史合作对'''
        return self.first_collab_year.get(pair, float('inf')) <= target_year

    def get_new_collab_pairs(self, target_year: int):
        '''获取指定年份的新增合作对'''
        new_pairs = [pair for pair, year in self.first_collab_year.items() if year == target_year]
        new_edges = self.all_edges[(self.all_edges['year'] == target_year) & (self.all_edges[['source', 'target']].apply(
                lambda x: tuple(sorted((x['source'], x['target']))), axis=1).isin(new_pairs))
        ]
        return new_edges

    def generate_candidate_pool(self, graph):
        '''生成目标年份的候选作者对 (+ 正样本，- 多策略负样本)'''
        target_year = graph.edata['year'].max().item()
        active_authors,author_id_to_idx,idx_to_author_id = self.generate_authorid2idx(target_year)
        target_df = self.df[self.df['publication_year'] == target_year]
        all_authors = list({aid for col in ['author1_id', 'author2_id', 'corresponding_author_ids'] for aid in target_df[col].dropna().unique()})
        pos_pairs = set(self.get_new_collab_pairs(target_year).apply(lambda x: tuple(sorted((x['source'], x['target']))), axis=1))

        sample_g = graph
        candidate_pool = []
        topo_candidates = set()
        valid_authors = [aid for aid in all_authors if aid in author_id_to_idx]
        # (1) 拓扑候选
        pos_list = [p[0] for p in pos_pairs]+[p[1] for p in pos_pairs] #获取正样本里所有的作者
        topo_candidates,semantic_candidates,random_candidates = set(),set(),set()
        for a in list(set(pos_list)): 
            a_id = torch.tensor(author_id_to_idx[a], device=self.device)
            _, hop1 = sample_g.out_edges(a_id, form='uv')
            #去重
            hop1 = hop1.unique()
            if len(hop1) == 0:
                continue
            #获取合作过的人合作过的人
            _, hop2 = sample_g.out_edges(hop1, form='uv')
            #去重
            hop2 = hop2.unique()
            if len(hop2) == 0:
                continue
            #前面两个去重
            mask = ~torch.isin(hop2, hop1)
            #得到最终的潜在合作候选人索引列表
            candidates_tensor = hop2[mask]
            if len(candidates_tensor) ==0:
                continue
            if len(candidates_tensor) == 1:
                candidates = [candidates_tensor.item()]
            else:
                candidates = candidates_tensor.tolist()
            #遍历生成合作对
            for dst_idx in candidates:
                dst = idx_to_author_id.get(dst_idx, None)
                if dst is None: continue
                pair = tuple(sorted((a, dst)))
                if pair not in pos_pairs and self._is_historical_new_pair(pair, target_year):
                    topo_candidates.add(pair)
         
            authors_pool = list(set([id for value in topo_candidates for id in value]))
            
            valid_author_embs = []
            for aid in authors_pool: # 对于拓扑候选池里的每一个作者
                emb = self._get_author_embedding(aid, target_year)  #获取文本嵌入
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                if emb.shape[-1] == 384:
                    valid_author_embs.append((aid, emb.to(self.device)))
            if len(valid_author_embs) >= 2:
                embeddings = [self.attention_pooling(emb) for _, emb in valid_author_embs]
                emb_matrix = torch.cat(embeddings, dim=0)
                sim_matrix = torch.cosine_similarity(emb_matrix.unsqueeze(1), emb_matrix.unsqueeze(0), dim=2)
                valid_author_ids = [aid for aid, _ in valid_author_embs]
                for i, aid in enumerate(valid_author_ids):
                    top_indices = sim_matrix[i].topk(2).indices
                    for j in top_indices:
                        j = j.item()
                        if i != j:
                            other = valid_author_ids[j]
                            pair = tuple(sorted((aid, other)))
                            if pair not in pos_pairs and self._is_historical_new_pair(pair, target_year):
                                semantic_candidates.add(pair)
        
        
        for a in list(set(pos_list)):
            max_attempts = 10
            for _ in range(max_attempts):
                try:
                    # 在全网络抽取正样本相当个数的随机负样本
                    pairs = random.sample(valid_authors, pos_list.count(a)) 
                    # 看看样本在不在正样本
                    valid_pairs = { tuple(sorted((a, b))) for b in pairs if a != b 
                                  and (tuple(sorted((a, b))) not in pos_pairs) and self._is_historical_new_pair((a, b), target_year)}
                    random_candidates.update(valid_pairs)
                    if len(random_candidates) >= pos_list.count(a): 
                        break
                except ValueError as e:
                    print(f"[!] 随机采样异常: {e} (authors: {len(valid_authors)})")
                    break
        
        
        candidate_pool = list(pos_pairs)+list(semantic_candidates) + list(random_candidates)
        labels = [1 if p in pos_pairs else 0 for p in candidate_pool]
        samples = pd.DataFrame({'source': [p[0] for p in candidate_pool], 'target': [p[1] for p in candidate_pool], 
                             'year': target_year,'label': labels})
        source_idx = torch.tensor(samples['source'].map(author_id_to_idx).values, dtype=torch.long)
        target_idx = torch.tensor(samples['target'].map(author_id_to_idx).values, dtype=torch.long)
        labels = torch.tensor(samples['label'].values, dtype=torch.float32)
        sample_dataset = torch.utils.data.TensorDataset(source_idx, target_idx, labels) 
        return sample_dataset

    def get_raw_data(self, author_id: str) -> dict: 
        """安全获取CPU端原始数据"""
        return self.author_meta.get(author_id, {
            'raw_text': '',
            'raw_papers': [],
            'raw_citations': 0,
            'raw_paper_count': 0
        })