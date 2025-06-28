"""
注意力池化机制

实现自注意力池化，用于将变长序列池化为固定长度的表示向量。
相比简单的平均池化或最大池化，注意力池化能更好地捕获重要信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AttentionPooling(nn.Module):
    """
    自注意力池化层
    
    将变长序列通过注意力机制池化为固定长度的向量表示。
    支持掩码处理，忽略padding位置。
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        attention_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_tanh: bool = True
    ):
        """
        初始化注意力池化层
        
        Args:
            hidden_dim: 输入隐藏维度
            attention_dim: 注意力计算维度，如果为None则使用hidden_dim
            dropout: Dropout概率
            use_tanh: 是否在注意力计算中使用tanh激活
        """
        super(AttentionPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        self.use_tanh = use_tanh
        
        # 注意力计算网络
        self.attention_linear = nn.Linear(hidden_dim, self.attention_dim)
        self.attention_vector = nn.Parameter(torch.randn(self.attention_dim))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.attention_linear.weight)
        nn.init.zeros_(self.attention_linear.bias)
        nn.init.normal_(self.attention_vector, std=0.1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入序列 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]，1表示有效位置，0表示padding
            
        Returns:
            池化后的表示 [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 计算注意力分数
        # [batch_size, seq_len, attention_dim]
        attention_hidden = self.attention_linear(hidden_states)
        
        if self.use_tanh:
            attention_hidden = torch.tanh(attention_hidden)
            
        # [batch_size, seq_len]
        attention_scores = torch.matmul(attention_hidden, self.attention_vector)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 将padding位置设为很小的值
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, -1e9
            )
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权平均
        # [batch_size, hidden_dim]
        pooled_output = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), 
            dim=1
        )
        
        return pooled_output
    
    def get_attention_weights(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可解释性分析）
        
        Args:
            hidden_states: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            注意力权重 [batch_size, seq_len]
        """
        with torch.no_grad():
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # 计算注意力分数
            attention_hidden = self.attention_linear(hidden_states)
            if self.use_tanh:
                attention_hidden = torch.tanh(attention_hidden)
            attention_scores = torch.matmul(attention_hidden, self.attention_vector)
            
            # 应用注意力掩码
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask == 0, -1e9
                )
            
            # 计算注意力权重
            attention_weights = F.softmax(attention_scores, dim=1)
            
        return attention_weights


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化层
    
    使用多个注意力头来捕获不同方面的信息，然后融合结果。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        attention_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        初始化多头注意力池化层
        
        Args:
            hidden_dim: 输入隐藏维度
            num_heads: 注意力头数量
            attention_dim: 每个头的注意力维度
            dropout: Dropout概率
        """
        super(MultiHeadAttentionPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim or (hidden_dim // num_heads)
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.head_dim = hidden_dim // num_heads
        
        # 多头注意力层
        self.attention_heads = nn.ModuleList([
            AttentionPooling(hidden_dim, self.attention_dim, dropout)
            for _ in range(num_heads)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            池化后的表示
        """
        batch_size = hidden_states.shape[0]
        
        # 将输入分割为多个头
        # [batch_size, seq_len, num_heads, head_dim]
        head_hidden = hidden_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        
        # 每个头进行注意力池化
        head_outputs = []
        for i, attention_head in enumerate(self.attention_heads):
            # [batch_size, seq_len, head_dim]
            head_input = head_hidden[:, :, i, :]
            head_output = attention_head(head_input, attention_mask)
            head_outputs.append(head_output)
        
        # 连接所有头的输出
        # [batch_size, hidden_dim]
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 输出投影
        output = self.output_projection(concatenated)
        output = self.dropout(output)
        
        return output


class HierarchicalAttentionPooling(nn.Module):
    """
    层次化注意力池化
    
    先在局部窗口内进行注意力池化，然后在全局进行注意力池化。
    适用于长文本序列的处理。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        window_size: int = 32,
        attention_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        初始化层次化注意力池化
        
        Args:
            hidden_dim: 隐藏维度
            window_size: 局部窗口大小
            attention_dim: 注意力计算维度
            dropout: Dropout概率
        """
        super(HierarchicalAttentionPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # 局部注意力池化
        self.local_attention = AttentionPooling(
            hidden_dim, attention_dim, dropout
        )
        
        # 全局注意力池化
        self.global_attention = AttentionPooling(
            hidden_dim, attention_dim, dropout
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入序列 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            池化后的表示 [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 计算需要的窗口数量
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        
        # 局部注意力池化
        window_representations = []
        for i in range(num_windows):
            start_idx = i * self.window_size
            end_idx = min((i + 1) * self.window_size, seq_len)
            
            # 提取窗口内容
            window_hidden = hidden_states[:, start_idx:end_idx, :]
            window_mask = None
            if attention_mask is not None:
                window_mask = attention_mask[:, start_idx:end_idx]
            
            # 局部池化
            window_repr = self.local_attention(window_hidden, window_mask)
            window_representations.append(window_repr)
        
        # 堆叠窗口表示
        # [batch_size, num_windows, hidden_dim]
        stacked_windows = torch.stack(window_representations, dim=1)
        
        # 全局注意力池化
        global_repr = self.global_attention(stacked_windows)
        
        return global_repr


class AdaptiveAttentionPooling(nn.Module):
    """
    自适应注意力池化
    
    根据输入序列的特点自适应地调整注意力计算方式。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        min_attention_dim: int = 64,
        max_attention_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化自适应注意力池化
        
        Args:
            hidden_dim: 隐藏维度
            min_attention_dim: 最小注意力维度
            max_attention_dim: 最大注意力维度
            dropout: Dropout概率
        """
        super(AdaptiveAttentionPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.min_attention_dim = min_attention_dim
        self.max_attention_dim = max_attention_dim
        
        # 自适应注意力维度预测网络
        self.dim_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 多个不同维度的注意力层
        self.attention_layers = nn.ModuleDict({
            str(dim): AttentionPooling(hidden_dim, dim, dropout)
            for dim in [min_attention_dim, 
                       (min_attention_dim + max_attention_dim) // 2,
                       max_attention_dim]
        })
        
        # 融合网络
        self.fusion_layer = nn.Linear(len(self.attention_layers) * hidden_dim, hidden_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            池化后的表示
        """
        # 计算序列的平均表示用于维度预测
        if attention_mask is not None:
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1).float()
            seq_repr = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        else:
            seq_repr = hidden_states.mean(dim=1)
        
        # 预测最适合的注意力维度权重
        dim_weights = self.dim_predictor(seq_repr)  # [batch_size, 1]
        
        # 使用不同维度的注意力层
        pooled_outputs = []
        for attention_layer in self.attention_layers.values():
            pooled = attention_layer(hidden_states, attention_mask)
            pooled_outputs.append(pooled)
        
        # 连接所有输出
        concatenated = torch.cat(pooled_outputs, dim=-1)
        
        # 融合
        fused_output = self.fusion_layer(concatenated)
        
        return fused_output


def create_attention_pooling(pooling_type: str, **kwargs) -> nn.Module:
    """
    创建注意力池化层的工厂函数
    
    Args:
        pooling_type: 池化类型 ('simple', 'multi_head', 'hierarchical', 'adaptive')
        **kwargs: 其他参数
        
    Returns:
        注意力池化层实例
    """
    pooling_classes = {
        'simple': AttentionPooling,
        'multi_head': MultiHeadAttentionPooling,
        'hierarchical': HierarchicalAttentionPooling,
        'adaptive': AdaptiveAttentionPooling
    }
    
    if pooling_type not in pooling_classes:
        raise ValueError(f"不支持的池化类型: {pooling_type}")
    
    return pooling_classes[pooling_type](**kwargs)


if __name__ == "__main__":
    # 测试代码
    batch_size, seq_len, hidden_dim = 4, 50, 128
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    # 模拟padding
    attention_mask[:, 40:] = 0
    
    # 测试简单注意力池化
    simple_pooling = AttentionPooling(hidden_dim)
    output1 = simple_pooling(hidden_states, attention_mask)
    print(f"简单注意力池化输出shape: {output1.shape}")
    
    # 测试多头注意力池化
    multi_head_pooling = MultiHeadAttentionPooling(hidden_dim, num_heads=8)
    output2 = multi_head_pooling(hidden_states, attention_mask)
    print(f"多头注意力池化输出shape: {output2.shape}")
    
    # 测试层次化注意力池化
    hierarchical_pooling = HierarchicalAttentionPooling(hidden_dim, window_size=16)
    output3 = hierarchical_pooling(hidden_states, attention_mask)
    print(f"层次化注意力池化输出shape: {output3.shape}")
    
    # 测试自适应注意力池化
    adaptive_pooling = AdaptiveAttentionPooling(hidden_dim)
    output4 = adaptive_pooling(hidden_states, attention_mask)
    print(f"自适应注意力池化输出shape: {output4.shape}")
    
    # 测试注意力权重获取
    attention_weights = simple_pooling.get_attention_weights(hidden_states, attention_mask)
    print(f"注意力权重shape: {attention_weights.shape}")
    print(f"注意力权重和: {attention_weights.sum(dim=1)}")  # 应该接近1
