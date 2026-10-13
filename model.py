import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

ROW = 16
COLUMN = 30
EMBED_DIM = 128
WINDOW_SHAPE = (4,6)
STATE_SPACE_SIZE = 10
'''
Modified Swin Transformer   (https://arxiv.org/pdf/2103.14030)
Input of the map should be in shape (16,30)
Which means in general the input tensor has shape (N, 16, 30, EMBED_DIM) = (N, 16, 30, 128)

Remarks:
10-13: I have implemented the masking of swmsa but it is not optimzed. I think we can try to ignore the softmax calculation for the masked windows.
10-13: There is also one flaw has to be fixed. The masking in the MHA_for_SWMSA module does not work for case n != 1. This can be fixed using reshape.
        But the problem is that calling reshape will need extra computation. And I suppose n will not be larger than 1 during training.
'''
class Multihead_Attention(nn.Module):
    '''Classical multihead attention \n
    Input should be in shape (n,l,embed_dim)
    '''
    def __init__(self,embed_dim:int,num_head:int=4):
        super().__init__()
        if embed_dim % num_head != 0:
            raise ValueError("Embedding dimension is not divisible by number of head")

        self.embed_dim = embed_dim
        self.h = num_head
        self.Q = nn.Linear(embed_dim,embed_dim)
        self.K = nn.Linear(embed_dim,embed_dim)
        self.V = nn.Linear(embed_dim,embed_dim)

    def split(self,X:torch.Tensor):
        n,l,d = X.shape
        return X.reshape(n,l,self.h,d // self.h).transpose(-2,-3)

    def forward(self,X:torch.Tensor,mask:torch.tensor=None,relative_position_bias:torch.Tensor = None):
        n,l,embed = X.shape

        #Q, K, V in shape (n,h,l,d)
        Q = self.split(self.Q(X))
        K = self.split(self.K(X))
        V = self.split(self.V(X))
        d = Q.shape[-1]

        if mask is None:
            attn_score = Q @ K.transpose(-1,-2)
        else:
            attn_score = Q @ K.transpose(-1,-2) * mask

        if relative_position_bias is None:
            z = F.softmax(attn_score / d ** 0.5,dim=-1) @ V    
        else:
            z = F.softmax(attn_score / d ** 0.5 + relative_position_bias,dim=-1) @ V    

        return z.transpose(-2,-3).reshape(n,l,embed)

class MHA_for_SWMSA(Multihead_Attention):
    def __init__(self, embed_dim, num_head = 4):
        super().__init__(embed_dim, num_head)

    def forward(self,X:torch.Tensor,mask:torch.tensor=None,relative_position_bias:torch.Tensor = None):
        '''mask in shape (n * num_window) \n
        For the incomplete windows(windows with elements have to be masked), we discard the calculation of the softmax output.
        '''
        n,l,embed = X.shape

        #Q, K, V in shape (n,h,l,d)
        Q = self.split(self.Q(X))
        K = self.split(self.K(X))
        V = self.split(self.V(X))
        d = Q.shape[-1]

        attn_score = Q @ K.transpose(-1,-2)
        if relative_position_bias is None:
            z = F.softmax(attn_score / d ** 0.5,dim=-1) @ V    
        else:
            z = F.softmax(attn_score / d ** 0.5 + relative_position_bias,dim=-1) @ V    

        if mask is not None:
            #z in shape (n * num_window, window_size, embed_dim)
            #mask in shape (n * num_window, 1, 1)
            z = z.masked_fill(mask,0)
            
        return z.transpose(-2,-3).reshape(n,l,embed)

class Relative_Position_Bias(nn.Module):
    def __init__(self,row:int,column:int,num_head:int):
        super().__init__()
        self.row = 2 * row - 1
        self.column = 2 * column - 1
        self.h = num_head
        self.B_head = nn.Parameter(torch.randn((num_head,self.row, self.column)))
        

    def get_by_distance(self,dr:int,dc:int,head:int):
        row = dr + ((self.row - 1) // 2)
        column = dc + ((self.column - 1) // 2)
        return self.get_by_index(row,column,head)

    def get_by_index(self,row:int,column:int,head:int):
        return self.B_head[head,row,column].item()

    def construct_index(self,M:torch.Tensor):
        _M = M.clone()
        r,c,_ = _M.shape
        _M[:,:,0] = _M[:,:,0] + ((self.row - 1) // 2)
        _M[:,:,1] = _M[:,:,1] + ((self.column - 1) // 2)
        return _M

    def construct_by_index(self,M:torch.Tensor):
        return self.B_head[:,M[:,:,0],M[:,:,1]]

class W_MSA(MHA_for_SWMSA):
    '''window-based multi-head self-attention \n
    Input should be in shape (n,r,c,embed_dim)

    :param window_size: The size of the window
    '''
    def __init__(self,embed_dim:int=EMBED_DIM,num_head:int=4,input_row:int=ROW,input_column:int=COLUMN,window_shape:tuple=WINDOW_SHAPE):
        super().__init__(embed_dim=embed_dim,num_head=num_head)
        if input_row  % window_shape[0] != 0 or input_column % window_shape[1] != 0:
            raise ValueError("Input shape is not divisible by window shape")

        self.window_shape = window_shape        
        self.r = input_row
        self.c = input_column

        wr,wc = window_shape
        self.rpb = Relative_Position_Bias(wr,wc,num_head)
        _B = torch.tensor(
            [[[(j // wc) - (i // wc) if k == 0 else (j % wc) - (i % wc) for k in range(2)]
                for j in range(wr * wc)] 
                for i in range(wr * wc)],
            dtype=torch.long)
        self.B_index = self.rpb.construct_index(_B)
        
    def _window_split(self, X):
        n, r, c, d = X.shape
        wr, wc = self.window_shape
        # (n, r//wr, wr, c//wc, wc, d)
        X = X.reshape(n, r // wr, wr, c // wc, wc, d)
        # (n, r//wr, c//wc, wr, wc, d) 
        X = X.permute(0, 1, 3, 2, 4, 5).contiguous()
        # (n, num_windows, wr*wc, d)
        X = X.reshape(n, (r // wr) * (c // wc), wr * wc, d)
        return X

    def _inverse_window_split(self, X):
        n, num_windows, window_size, d = X.shape
        wr, wc = self.window_shape
        r, c = self.r, self.c
        # (n, r//wr, c//wc, wr, wc, d)
        X = X.reshape(n, r // wr, c // wc, wr, wc, d)
        # (n, r//wr, wr, c//wc, wc, d)
        X = X.permute(0, 1, 3, 2, 4, 5).contiguous()
        # (n, r, c, d)
        X = X.reshape(n, r, c, d)
        return X

    def forward(self, X:torch.Tensor, mask:torch.Tensor=None):
        n, r, c, d = X.shape
        X_win = self._window_split(X)                
        n, num_windows, ws, d = X_win.shape

        X_flat = X_win.reshape(n * num_windows, ws, d)
        out = super().forward(X_flat,mask,self.rpb.construct_by_index(self.B_index)).reshape(n, num_windows, ws, d)                
        return self._inverse_window_split(out)

class SW_MSA(W_MSA):
    def __init__(self, embed_dim:int = EMBED_DIM, num_head:int = 4, input_row:int = ROW, input_column:int = COLUMN, window_shape:tuple = WINDOW_SHAPE):
        super().__init__(embed_dim, num_head, input_row, input_column, window_shape)
        wr,wc = window_shape
        self.shift_row = wr // 2
        self.shift_column = wc // 2

        self.rolled_mask = torch.tensor([[True if i == 0 or j == 0 else False for j in range(self.c // wc)] for i in range(self.r // wr)]).reshape(-1,1,1,1)

    def _roll(self,X):
        n, r, c, d = X.shape
        
        X = torch.roll(X,self.shift_row,dims=1)
        X = torch.roll(X,self.shift_column,dims=2)

        return X

    def forward(self, X:torch.Tensor):
        X = self._roll(X)
        return super().forward(X,self.rolled_mask)


class Swin_Transformer_Blcok(nn.Module): 
    '''Swin transformer encoder block
    '''
    def __init__(self,input_row:int=ROW,input_column:int=COLUMN,window_shape=WINDOW_SHAPE,embed_dim:int=EMBED_DIM,dropout_prob:float=0.5,attn_num_head:int=4):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.layernorm4 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.dropout4 = nn.Dropout(dropout_prob)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim,embed_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim,embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim,embed_dim)
        )

        self.wmsa = W_MSA(embed_dim,attn_num_head,input_row,input_column,window_shape)
        self.swmsa = SW_MSA(embed_dim,attn_num_head,input_row,input_column,window_shape)
        
    def forward(self,X):
        X1 = self.dropout1(self.layernorm1(self.wmsa(X))) + X
        X2 = self.dropout2(self.layernorm2(self.mlp1(X1))) + X1 
        X3 = self.dropout3(self.layernorm3(self.swmsa(X2))) + X2
        X4 = self.dropout4(self.layernorm4(self.mlp2(X3))) + X3

        return X4

class Shifted_Window_Attn_ViT(nn.Module):
    def __init__(self,input_row:int=ROW,input_column:int=COLUMN,window_shape:tuple=WINDOW_SHAPE,embed_dim:int=EMBED_DIM,dropout_prob:float=0.5,attn_num_head:int=4):
        super().__init__()
        self.block1 = Swin_Transformer_Blcok(input_row,input_column,window_shape,embed_dim,dropout_prob,attn_num_head)
        self.block2 = Swin_Transformer_Blcok(input_row,input_column,window_shape,embed_dim,dropout_prob,attn_num_head)    
        self.block3 = Swin_Transformer_Blcok(input_row,input_column,window_shape,embed_dim,dropout_prob,attn_num_head)
        self.block4 = Swin_Transformer_Blcok(input_row,input_column,window_shape,embed_dim,dropout_prob,attn_num_head)

        self.embedding = nn.Embedding(STATE_SPACE_SIZE,embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim,2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim,4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, 1)
        )

    def forward(self,X):
        X_embed = self.embedding(X)
        X1 = self.block1(X_embed)
        X2 = self.block2(X1)
        X3 = self.block2(X2)
        X4 = self.block2(X3)

        return self.output(X4)