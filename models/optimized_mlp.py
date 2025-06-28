"""
优化的多层感知机文本分类模型

结合注意力机制、批归一化、dropout等技术，专门针对小数据集优化的MLP模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import math

from .attention_pooling import AttentionPooling


class OptimizedTextMLP(nn.Module):
    """
    优化的文本分类MLP模型
    
    特性:
    - 可训练的词嵌入层
    - 注意力池化机制
    - 批归一化和Dropout正则化
    - 灵活的隐藏层配置
    - 标签平滑损失支持
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 3,
        dropout_rate: float = 0.5,
        padding_idx: int = 0,
        use_attention_pooling: bool = True,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        初始化优化MLP模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类类别数
            dropout_rate: Dropout概率
            padding_idx: 填充标记的索引
            use_attention_pooling: 是否使用注意力池化
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型
        """
        super(OptimizedTextMLP, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_attention_pooling = use_attention_pooling
        self.use_batch_norm = use_batch_norm
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # 池化层
        if use_attention_pooling:
            self.pooling = AttentionPooling(embedding_dim)
        else:
            self.pooling = None
            
        # 激活函数
        self.activation = self._get_activation(activation)
        
        # 构建MLP层
        self.mlp_layers = self._build_mlp_layers(
            embedding_dim, hidden_dims, num_classes
        )
        
        # 初始化参数
        self._init_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1)
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _build_mlp_layers(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int
    ) -> nn.ModuleList:
        """构建MLP层"""
        layers = nn.ModuleList()
        
        # 输入维度
        current_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            # 线性层
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # 批归一化
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            # 激活函数
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            
            current_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        return layers
    
    def _init_weights(self):
        """初始化模型参数"""
        # 初始化嵌入层
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        if hasattr(self.embedding, 'padding_idx'):
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
        
        # 初始化线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        # 词嵌入 [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # 池化操作
        if self.use_attention_pooling and self.pooling is not None:
            # 注意力池化
            pooled = self.pooling(embedded, attention_mask)
        else:
            # 平均池化（忽略padding）
            if attention_mask is not None:
                # 应用掩码
                embedded = embedded * attention_mask.unsqueeze(-1).float()
                pooled = embedded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
            else:
                pooled = embedded.mean(dim=1)
        
        # 通过MLP层
        output = pooled
        for layer in self.mlp_layers:
            output = layer(output)
            
        return output
    
    def get_embeddings(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取文本表示向量（最后一个隐藏层的输出）
        
        Args:
            x: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            文本表示向量
        """
        # 嵌入和池化
        embedded = self.embedding(x)
        
        if self.use_attention_pooling and self.pooling is not None:
            pooled = self.pooling(embedded, attention_mask)
        else:
            if attention_mask is not None:
                embedded = embedded * attention_mask.unsqueeze(-1).float()
                pooled = embedded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
            else:
                pooled = embedded.mean(dim=1)
        
        # 通过除了最后一层的所有MLP层
        output = pooled
        for layer in self.mlp_layers[:-1]:  # 排除最后的输出层
            output = layer(output)
            
        return output
    
    def predict_proba(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测概率分布
        
        Args:
            x: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            类别概率分布
        """
        with torch.no_grad():
            logits = self.forward(x, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def predict(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            预测类别索引
        """
        probabilities = self.predict_proba(x, attention_mask)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        获取注意力权重（用于可解释性分析）
        
        Args:
            x: 输入序列
            attention_mask: 注意力掩码
            
        Returns:
            注意力权重，如果未使用注意力池化则返回None
        """
        if not self.use_attention_pooling or self.pooling is None:
            return None
            
        embedded = self.embedding(x)
        attention_weights = self.pooling.get_attention_weights(embedded, attention_mask)
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "OptimizedTextMLP",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "use_attention_pooling": self.use_attention_pooling,
            "use_batch_norm": self.use_batch_norm,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设float32
        }
    
    def freeze_embedding(self):
        """冻结嵌入层参数"""
        for param in self.embedding.parameters():
            param.requires_grad = False
            
    def unfreeze_embedding(self):
        """解冻嵌入层参数"""
        for param in self.embedding.parameters():
            param.requires_grad = True


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    用于提升模型在小数据集上的泛化能力
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        """
        初始化标签平滑损失
        
        Args:
            smoothing: 平滑参数 (0-1)
            ignore_index: 忽略的标签索引
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑交叉熵损失
        
        Args:
            pred: 预测logits [batch_size, num_classes]
            target: 真实标签 [batch_size]
            
        Returns:
            损失值
        """
        batch_size, num_classes = pred.shape
        
        # 过滤忽略的标签
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            target = target[mask]
            pred = pred[mask]
            
            if target.numel() == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 标签平滑
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算交叉熵
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()
        
        return loss


def create_model_from_config(config: Dict[str, Any]) -> OptimizedTextMLP:
    """
    从配置创建模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        创建的模型实例
    """
    model_config = config.get("model", {})
    
    return OptimizedTextMLP(
        vocab_size=model_config.get("embedding", {}).get("vocab_size", 15000),
        embedding_dim=model_config.get("embedding", {}).get("embedding_dim", 128),
        hidden_dims=model_config.get("mlp", {}).get("hidden_dims", [256, 128]),
        num_classes=model_config.get("num_classes", 3),
        dropout_rate=model_config.get("mlp", {}).get("dropout_rate", 0.5),
        padding_idx=model_config.get("embedding", {}).get("padding_idx", 0),
        use_attention_pooling=model_config.get("attention", {}).get("use_attention_pooling", True),
        use_batch_norm=model_config.get("mlp", {}).get("use_batch_norm", True),
        activation=model_config.get("mlp", {}).get("activation", "relu")
    )


if __name__ == "__main__":
    # 测试模型
    model = OptimizedTextMLP(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dims=[256, 128],
        num_classes=3
    )
    
    # 打印模型信息
    print(model.get_model_info())
    
    # 测试前向传播
    batch_size, seq_len = 4, 50
    x = torch.randint(0, 10000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = model(x, attention_mask)
    print(f"Output shape: {output.shape}")
    
    # 测试预测功能
    predictions = model.predict(x, attention_mask)
    probabilities = model.predict_proba(x, attention_mask)
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # 测试注意力权重
    attention_weights = model.get_attention_weights(x, attention_mask)
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    # 测试标签平滑损失
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    targets = torch.randint(0, 3, (batch_size,))
    loss = criterion(output, targets)
    print(f"Label smoothing loss: {loss.item()}")
