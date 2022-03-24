import torch.nn as nn
import torch.nn.functional as F
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self,d_model,dk,heads,d_ff,nx,alpha):
        super(Encoder, self).__init__()
        #重复模型块用nn.ModuleList，参数是一个[x for i in range(t)],但是没有实现forward功能；Sequential要确保输入输入维度匹配，实现了forward功能
        # self.encoder = torch.nn.ModuleList([EncoderLayer(d_model,dk,heads_1+heads_2,d_ff) for _ in range(nx)])
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, dk, heads,alpha) for _ in range(nx)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,input,mask,relation):
        enc_self_attns = []
        for encoder_layer in self.layers:
            input,atts = encoder_layer(input,mask,relation)
            enc_self_attns.append(atts)
        return self.norm(input), enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(self,d_model,dk,heads,alpha):
        super(EncoderLayer, self).__init__()
        self.attLayer = AttSubLayer(d_model,dk,heads)
        self.attLayerv2 = AttSubLayerv2(d_model,dk,heads)
        self.ratLayer = ratAttSublayer(d_model,dk,heads)
        self.feedLayer = PositionwiseFeedForward(d_model, 4*d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.heads_1 = 4
        self.heads_2 = 0
        self.dataflowLayer = adjAttSubLayer(d_model,dk,self.heads_1,self.heads_2,alpha)
    def forward(self,input,mask,relation):
        # att_output,att_matrix = self.adjAttSubLayer(input,input,input,mask,relation)
        # att_output, att_matrix = self.attLayer(input, input, input, mask)
        backup = input
        #这里的两个norm操作是仿照annotated transformer，就和relation机制一样
        input = self.norm(input)
        att_output, att_matrix = self.attLayerv2(input, input, input, mask,relation)
        att_output = backup + self.dropout(att_output)
        feed_input = self.norm(att_output)
        output = self.feedLayer(feed_input)
        output = att_output + self.dropout(output)
        return output,att_matrix

class AttSubLayer(nn.Module):
    def __init__(self,d_model,dk,heads):
        super(AttSubLayer, self).__init__()
        self.dk = torch.tensor(dk)
        self.heads = heads
        self.d_model = d_model
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d_model,dk*heads,bias=False) for _ in range(3)])
        self.fc = torch.nn.Linear(dk * heads, d_model)
        self.norm = torch.nn.LayerNorm(self.d_model)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self,query,key,value,mask):
        batch_size = query.shape[0]
        Q,K,V = [l(x).reshape(batch_size,-1,self.heads,self.dk).transpose(2,1) for l,x in zip(self.linears, (query,key,value))]
        scores = torch.matmul(Q,K.transpose(-1,-2)) / torch.sqrt(self.dk)
        mask = mask.unsqueeze(1).repeat(1,self.heads, 1,1).bool()
        scores.masked_fill_(mask,-1e9)
        scores = self.dropout(torch.softmax(scores, dim=-1))
        att = torch.matmul(scores,V).transpose(2,1).reshape(batch_size, -1, self.heads*self.dk)
        att = self.fc(att)
        return self.norm(att + query),scores

class AttSubLayerv2(nn.Module):
    def __init__(self,d_model,dk,heads):
        super(AttSubLayerv2, self).__init__()
        self.dk = torch.tensor(dk)
        self.heads = heads
        self.d_model = d_model
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d_model,dk*heads,bias=False) for _ in range(3)])
        self.fc = torch.nn.Linear(dk * heads, d_model)
        self.norm = torch.nn.LayerNorm(self.d_model)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self,query,key,value,mask,relation):
        batch_size = query.shape[0]
        Q,K,V = [l(x).reshape(batch_size,-1,self.heads,self.dk).transpose(2,1) for l,x in zip(self.linears, (query,key,value))]
        relation = relation.unsqueeze(1)
        scores = torch.matmul(torch.matmul(Q,K.transpose(-1,-2)),relation) / torch.sqrt(self.dk)
        mask = mask.unsqueeze(1).repeat(1,self.heads, 1,1).bool()
        scores.masked_fill_(mask,-1e9)
        scores = self.dropout(torch.softmax(scores, dim=-1))
        att = torch.matmul(scores,V).transpose(2,1).reshape(batch_size, -1, self.heads*self.dk)
        att = self.fc(att)
        return self.norm(att + query),scores

class ratAttSublayer(nn.Module):
    def __init__(self,d_model,dk,heads):
        super(ratAttSublayer, self).__init__()
        self.dk = torch.tensor(dk)
        self.heads = heads
        self.d_model = d_model
        self.num_relation_kinds = 16
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d_model,dk*heads,bias=False) for _ in range(3)])
        self.fc = torch.nn.Linear(dk * heads, d_model)
        self.relation_k_emb = nn.Embedding(self.num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(self.num_relation_kinds, self.self_attn.d_k)

    def forward(self,query,key,value,mask,relation):
        batch_size = query.shape[0]
        Q,K,V = [l(x).reshape(batch_size,-1,self.heads,self.dk).transpose(2,1) for l,x in zip(self.linears, (query,key,value))]
        relation = relation.unsqueeze(1)
        scores = torch.matmul(Q, K.transpose(-1,-2)) + torch.matmul(Q, relation)
        scores = scores / torch.sqrt(self.dk)
        mask = mask.unsqueeze(1).repeat(1,self.heads, 1,1)
        #给true的地方赋值-1e9
        scores = scores.masked_fill_(mask==0,-1e9)
        scores = torch.softmax(scores, dim=-1)
        att = torch.matmul(scores,V).transpose(2,1) + torch.matmul(scores, relation)
        att = att.reshape(batch_size, -1, self.heads*self.dk)
        att = self.fc(att)
        return att, scores

class adjAttSubLayer(nn.Module):
    def __init__(self,d_model,dk,heads_1,heads_2,alpha):
        super(adjAttSubLayer, self).__init__()
        self.dk = torch.tensor(dk)
        self.heads_1 = heads_1
        self.heads_2 = heads_2
        self.heads = heads_1+heads_2
        self.alpha = alpha
        self.d_model = d_model
        self.linears_1 = torch.nn.ModuleList([torch.nn.Linear(d_model,dk*heads_1,bias=False) for _ in range(3)])
        self.linears_2 = torch.nn.ModuleList([torch.nn.Linear(d_model,dk*heads_2,bias=False) for _ in range(3)])
        self.fc = torch.nn.Linear(dk * (self.heads), d_model)
        self.norm = torch.nn.LayerNorm(self.d_model)
    def forward(self,query,key,value,mask,relation):
        backup = query
        batch_size = query.shape[0]
        Q1,K1,V1 = [l(x).reshape(batch_size,-1,self.heads_1,self.dk).transpose(2,1) for l,x in zip(self.linears, (query,key,value))]
        Q2,K2,V2 = [l(x).reshape(batch_size,-1,self.heads_2,self.dk).transpose(2,1) for l,x in zip(self.linears, (query,key,value))]

        #relation的维度肯定和Q*K的不一致，因为后者有heads作为第二维度
        relation = relation.unsqueeze(1).repeat(1, self.heads_1, 1,1)
        # print("repeat is right",relation[0][0] == relation[0][1] , relation[0][2]  == relation[0][3])
        scores_1 = torch.matmul(Q1,K1.transpose(-1,-2)) / torch.sqrt(self.dk) + self.alpha * relation

        # scores_2 = torch.matmul(Q2,K2.transpose(-1,-2))

        #torch.Size([10, 2, 49, 49]) torch.Size([10, 49, 49])
        # print("fault happend",scores_2.shape, relation.shape)
        # scores_2 = (scores_2 + self.alpha * torch.matmul(scores_2, relation)) / torch.sqrt(self.dk)

        # print(self.heads,mask.shape)
        # mask = mask.unsqueeze(1).repeat(1,self.heads, 1,1).bool()

        mask_1 = mask.unsqueeze(1).repeat(1,self.heads_1, 1,1).bool()
        # mask_2 = mask.unsqueeze(1).repeat(1,self.heads_2, 1,1).bool()

        # print("mask1",mask_1.shape,mask_1)
        scores_1.masked_fill_(mask_1,-1e9)
        # scores_2.masked_fill_(mask_2, -1e9)

        # print(scores)
        scores_1 = torch.softmax(scores_1, dim=-1)
        # scores_2 = torch.softmax(scores_2, dim=-1)
        # scores = torch.cat((scores_1,scores_2),1)

        att_1 = torch.matmul(scores_1,V1).transpose(2,1).reshape(batch_size, -1, self.heads_1*self.dk)
        # att_2 = torch.matmul(scores_2,V2).transpose(2,1).reshape(batch_size, -1, self.heads_2*self.dk)

        # print("seee",att_1.shape,att_2.shape)
        # att = torch.cat((att_1,att_2),-1)
        att = self.fc(att_1)
        return self.norm(att + backup),scores_1