# LLM agent预测模块 Collaborationself
import re
import dgl
from ResearchGraphDataset import * 
from openai import OpenAI
from typing import Optional
from zhipuai import ZhipuAI


class CollaborationPredictorglm:
    def __init__(self, dataset: ResearchGraphDataset, data: dict):
        self.dataset = dataset
        self.device = dataset.device
        self.apikey = " "
        self.data = data
    
    
    def round1_1_prompt(self, graph: dgl.DGLGraph, target_pair: tuple) -> str:
        key = f'{target_pair}'
        try:
            self.data['a_papers_num'][f'{target_pair}']
        except:
            key = f'{tuple(reversed(target_pair))}'
    
        prompt_round1_1 = f'''
            假设你是编号为{eval(key)[0]}的科研人员，现在科研人员{eval(key)[1]}想要与你进行合作。请根据你的自身定位以及目标合作者的信息，对你们的科研合作潜力进行分析。你当前学术特征包括：
    - 研究主题：{self.data['a_text'][key][:300]}
    - 点度中心性: {self.data['a_degree_centrality'][key]:.5f}（反映你的网络影响力）
    - 结构洞约束系数{self.data['a_constraint'][key]:.5f}（值越低越可能成为桥梁节点）
    - 近三年发文数: {self.data['a_papers_num'][key]}
    - 总被引量: {self.data['a_citations'][key]:.1f}
你与对方的网络可达性：{self.data['dist'][key]}，你科研网络中2 hop的科研人员包括{[n for n in self.data['a_neighbors'][key] if n != eval(key)[0]]}，对方科研网络中2 hop的科研人员包括{[n for n in self.data['b_neighbors'][key] if n != eval(key)[1]]}。
对方的基本情况如下：
    - 研究主题：{self.data['b_text'][key][:300]}
    - 点度中心性: {self.data['b_degree_centrality'][key]:.5f}（反映你的网络影响力）
    - 结构洞约束系数{self.data['b_constraint'][key]:.5f}（值越低越可能成为桥梁节点）
    - 近三年发文数: {self.data['b_papers_num'][key]}
    - 总被引量: {self.data['b_citations'][key]:.1f}。
    可以用于分析合作经历的理论包括：1.社会交换理论：该理论认为交换具有双向性，个体之间随时间推移形成互惠交易和相互依赖关系。科研合作同样遵循上述理论的互惠原则，即科研合作者在制定合作策略时会考虑之前的结果；2.成本效益理论：该理论认为个体倾向于通过比较成本和收益来最大化自身效用。科研个体在决定是否继续协作时，可能会权衡长期收益与长期成本；3.邻近性原则，在合作网络中更靠近的个体合作可能性更大。请从你的学术特征出发，分析对方的绝对数据以及对方数据相对于你的差异，依照3个理论分析目标作者的合作潜力，输出分析报告。具体格式如下："分析结果":[文本]：范围0-100字，包含3个理论框架独立的分析结果。
            '''
        return prompt_round1_1

    def Zhipu_request(self,Prompt):
        while True:
            try:
                #print(Prompt)
                #print(f"Prompt length: {len(Prompt)} characters")
                client = ZhipuAI(api_key=" ")  
                response = client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[
            {"role": "user", "content": Prompt}])
                return response
            except Exception as e:
                print(f"发生异常: {e}")
                raise Exception
            
    def Zhipu_request_json(self,Prompt):
        while True:
            try:
                #print(Prompt)
                #print(f"Prompt length: {len(Prompt)} characters")
                client = ZhipuAI(api_key=" ")  
                response = client.chat.completions.create(
                    model="glm-4-plus",
                    messages=[{"role": "user", "content": Prompt}],
                    response_format={'type': 'json_object'})
                return response
            except Exception as e:
                print(f"发生异常: {e}")
                raise Exception
    
    def round1_2_prompt(self, target_pair, analysis: str) -> str:
        key = f'{target_pair}'
        try:
            self.data['a_degree_centrality'][f'{target_pair}']
        except:
            key = f'{tuple(reversed(target_pair))}'

        return f'''
    假设你是编号为{eval(key)[0]}的科研人员，现在科研人员{eval(key)[1]}想要与你进行合作。请根据你的自身定位以及目标合作者的信息，对你们的科研合作潜力进行分析。你当前学术特征包括：
        - 研究主题：{self.data['a_text'][key][:300]}
        - 点度中心性: {self.data['a_degree_centrality'][key]:.5f}（反映你的网络影响力）
        - 结构洞约束系数{self.data['a_constraint'][key]:.5f}（值越低越可能成为桥梁节点）
        - 近三年发文数: {self.data['a_papers_num'][key]}
        - 总被引量: {self.data['a_citations'][key]:.1f}
    你与对方的网络可达性：{self.data['dist'][key]}，你科研网络中2 hop的科研人员包括{[n for n in self.data['a_neighbors'][key] if n != eval(key)[0]]}，对方科研网络中2 hop的科研人员包括{[n for n in self.data['b_neighbors'][key] if n != eval(key)[1]]}。
    对方的基本情况如下：
        - 研究主题：{self.data['b_text'][key][:300]}
        - 点度中心性: {self.data['b_degree_centrality'][key]:.5f}（反映你的网络影响力）
        - 结构洞约束系数{self.data['b_constraint'][key]:.5f}（值越低越可能成为桥梁节点）
        - 近三年发文数: {self.data['b_papers_num'][key]}
        - 总被引量: {self.data['b_citations'][key]:.1f}。
        {analysis}上述分析是你对于其科研网络内两位作者合作潜力的分析报告。请基于你自身及对方的学术特征、网络特征以及分析报告，给出你与对方对进行合作的概率[0,1]。并生成30字以内的合作主题想法。
        输出格式包括：
        Prob:[数值]：范围[0-1]，小数点后两位。
        Theme:[文本]：范围[0-30]字。
        Reason:[文本]：范围[0-50]字，简要说明预测依据。
        '''
 
    def predict(self, target_pair: tuple, graph: Optional[dgl.DGLGraph] = None) -> dict:
        current_year = 2023
        round1_1_results = []

        prompt_round1_1 = self.round1_1_prompt(graph, target_pair) 
        analysis = self.Zhipu_request(prompt_round1_1).choices[0].message.content
        round1_1_results.append({'agent':target_pair[0], 'analysis': analysis})
        
        prompt_a = self.round1_2_prompt(target_pair,analysis) #A方
        decision_a = eval(self.Zhipu_request_json(prompt_a).choices[0].message.content)
        max_try = 0 
        while 'Prob' not in list(decision_a.keys()):
            decision_a = eval(self.Zhipu_request_json(prompt_a).choices[0].message.content)
            max_try = max_try + 1
            if max_try >10:
                raise Exception("这句死活出不来json的Prob")


        return {'final_prob': float(decision_a['Prob']),
                'agent': target_pair[0],
            'final_theme': decision_a['Theme'],
            'reason': decision_a['Reason'],
            'analysis': analysis}



        
















