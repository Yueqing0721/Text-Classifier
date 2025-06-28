"""
数据加载和预处理工具模块

提供统一的数据加载接口，支持多种数据格式和预处理选项。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import re
import jieba
from collections import Counter
import pickle
import logging
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化文本预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 预处理配置
        preprocessing_config = self.config.get("preprocessing", {})
        text_cleaning_config = preprocessing_config.get("text_cleaning", {})
        tokenization_config = preprocessing_config.get("tokenization", {})
        vocab_config = preprocessing_config.get("vocabulary", {})
        
        # 文本清洗参数
        self.remove_html = text_cleaning_config.get("remove_html", True)
        self.remove_urls = text_cleaning_config.get("remove_urls", True)
        self.remove_emails = text_cleaning_config.get("remove_emails", True)
        self.normalize_whitespace = text_cleaning_config.get("normalize_whitespace", True)
        self.min_length = text_cleaning_config.get("min_length", 50)
        self.max_length = text_cleaning_config.get("max_length", 5000)
        
        # 分词参数
        self.tokenization_method = tokenization_config.get("method", "whitespace")
        self.lowercase = tokenization_config.get("lowercase", True)
        self.remove_punctuation = tokenization_config.get("remove_punctuation", False)
        self.remove_stopwords = tokenization_config.get("remove_stopwords", False)
        
        # 词汇表参数
        self.min_freq = vocab_config.get("min_freq", 2)
        self.max_vocab_size = vocab_config.get("max_vocab_size", 15000)
        self.special_tokens = vocab_config.get("special_tokens", {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>"
        })
        
        # 初始化词汇表
        self.word_to_idx = {
            self.special_tokens["pad_token"]: 0,
            self.special_tokens["unk_token"]: 1
        }
        self.idx_to_word = {0: self.special_tokens["pad_token"], 1: self.special_tokens["unk_token"]}
        self.vocab_size = 2
        
        # 停用词（简单列表，实际应用中可以从文件加载）
        self.stopwords = set([
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"
        ]) if self.remove_stopwords else set()
        
        # 标点符号正则
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        
        # HTML标签正则
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # URL正则
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # 邮箱正则
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        if self.remove_html:
            text = self.html_pattern.sub('', text)
        
        # 移除URL
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # 移除邮箱
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # 转换为小写
        if self.lowercase:
            text = text.lower()
        
        # 移除标点符号
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        
        # 标准化空白字符
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 文本
            
        Returns:
            词汇列表
        """
        if self.tokenization_method == "jieba":
            # 使用jieba分词（适合中文）
            tokens = list(jieba.cut(text))
        elif self.tokenization_method == "whitespace":
            # 空白符分词
            tokens = text.split()
        else:
            # 默认空白符分词
            tokens = text.split()
        
        # 过滤停用词
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # 过滤短词和纯数字
        tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> int:
        """
        构建词汇表
        
        Args:
            texts: 文本列表
            
        Returns:
            词汇表大小
        """
        self.logger.info("开始构建词汇表...")
        
        # 统计词频
        word_freq = Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            word_freq.update(tokens)
        
        self.logger.info(f"原始词汇数量: {len(word_freq)}")
        
        # 按频率排序，取前max_vocab_size-2个（减去特殊tokens）
        most_common_words = word_freq.most_common(self.max_vocab_size - 2)
        
        # 添加到词汇表（频率大于min_freq的词汇）
        idx = 2  # 0: PAD, 1: UNK
        for word, freq in most_common_words:
            if freq >= self.min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        self.vocab_size = len(self.word_to_idx)
        self.logger.info(f"最终词汇表大小: {self.vocab_size}")
        
        return self.vocab_size
    
    def text_to_indices(self, text: str, max_length: int = None) -> List[int]:
        """
        将文本转换为索引序列
        
        Args:
            text: 文本
            max_length: 最大长度
            
        Returns:
            索引列表
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        # 转换为索引
        indices = []
        unk_idx = self.word_to_idx[self.special_tokens["unk_token"]]
        
        for token in tokens:
            idx = self.word_to_idx.get(token, unk_idx)
            indices.append(idx)
        
        # 截断或填充
        max_len = max_length or self.config.get("model", {}).get("embedding", {}).get("max_length", 256)
        pad_idx = self.word_to_idx[self.special_tokens["pad_token"]]
        
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices.extend([pad_idx] * (max_len - len(indices)))
        
        return indices
    
    def create_attention_mask(self, indices: List[int]) -> List[int]:
        """
        创建注意力掩码
        
        Args:
            indices: 索引列表
            
        Returns:
            注意力掩码
        """
        pad_idx = self.word_to_idx[self.special_tokens["pad_token"]]
        mask = [1 if idx != pad_idx else 0 for idx in indices]
        return mask
    
    def save_vocabulary(self, save_path: str):
        """保存词汇表"""
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocab_size": self.vocab_size,
            "config": self.config
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        self.logger.info(f"词汇表已保存: {save_path}")
    
    def load_vocabulary(self, load_path: str):
        """加载词汇表"""
        with open(load_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_idx = vocab_data["word_to_idx"]
        self.idx_to_word = vocab_data["idx_to_word"]
        self.vocab_size = vocab_data["vocab_size"]
        
        self.logger.info(f"词汇表已加载: {load_path}, 大小: {self.vocab_size}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocab_size": self.vocab_size,
            "preprocessing_config": self.config
        }


class NewsDataset(Dataset):
    """新闻数据集类"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        preprocessor: TextPreprocessor,
        class_names: List[str] = None,
        max_length: int = None
    ):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            preprocessor: 文本预处理器
            class_names: 类别名称列表
            max_length: 序列最大长度
        """
        self.data = data
        self.preprocessor = preprocessor
        self.class_names = class_names or ["政治/政府", "商业/经济", "科技/科学"]
        self.max_length = max_length or 256
        
        # 创建标签映射
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.class_names)}
        
        # 处理数据
        self.processed_data = self._process_data()
        
    def _process_data(self) -> List[Dict[str, Any]]:
        """处理数据"""
        processed = []
        
        for item in self.data:
            # 获取文本和标签
            text = item.get("content", "") or item.get("text", "")
            title = item.get("title", "")
            
            # 合并标题和内容
            if title and text:
                full_text = f"{title}. {text}"
            else:
                full_text = text or title
            
            # 获取标签
            label = item.get("label") or item.get("category")
            if isinstance(label, str) and label in self.label_to_idx:
                label_idx = self.label_to_idx[label]
            elif isinstance(label, int) and 0 <= label < len(self.class_names):
                label_idx = label
            else:
                # 如果没有标签或标签无效，跳过
                continue
            
            # 转换文本为索引
            indices = self.preprocessor.text_to_indices(full_text, self.max_length)
            attention_mask = self.preprocessor.create_attention_mask(indices)
            
            processed.append({
                "text": indices,
                "attention_mask": attention_mask,
                "label": label_idx,
                "original_text": full_text,
                "metadata": item
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_data[idx]
        
        return {
            "text": torch.tensor(item["text"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        label_counts = Counter(item["label"] for item in self.processed_data)
        return {self.class_names[label]: count for label, count in label_counts.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        text_lengths = [len([idx for idx in item["text"] if idx != 0]) for item in self.processed_data]
        
        return {
            "total_samples": len(self.processed_data),
            "class_distribution": self.get_class_distribution(),
            "text_length_stats": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "median": np.median(text_lengths)
            }
        }


class NewsDataLoader:
    """新闻数据加载器"""
    
    def __init__(self, data_dir: str = "data/", config: Dict[str, Any] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录
            config: 配置
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从单个文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.logger.info(f"从文件加载数据: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return self._load_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            return self._load_jsonl(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def load_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        从目录加载所有数据文件
        
        Args:
            directory: 目录路径
            
        Returns:
            合并的数据列表
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        self.logger.info(f"从目录加载数据: {directory}")
        
        all_data = []
        
        # 支持的文件格式
        supported_extensions = ['.json', '.csv', '.jsonl']
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    data = self.load_from_file(file_path)
                    all_data.extend(data)
                    self.logger.info(f"从 {file_path} 加载 {len(data)} 个样本")
                except Exception as e:
                    self.logger.error(f"加载文件失败 {file_path}: {e}")
        
        self.logger.info(f"总共加载 {len(all_data)} 个样本")
        return all_data
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON文件格式不正确")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载CSV文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    
    def create_train_val_test_split(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: str = "label",
        random_state: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        分割数据集
        
        Args:
            data: 原始数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            stratify_column: 分层依据的列名
            random_state: 随机种子
            
        Returns:
            训练集、验证集、测试集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        # 提取标签用于分层
        labels = [item.get(stratify_column) for item in data]
        
        # 首先分割出测试集
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=random_state
        )
        
        # 然后分割训练集和验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=random_state
        )
        
        self.logger.info(f"数据分割完成:")
        self.logger.info(f"  训练集: {len(train_data)} 样本")
        self.logger.info(f"  验证集: {len(val_data)} 样本")
        self.logger.info(f"  测试集: {len(test_data)} 样本")
        
        return train_data, val_data, test_data
    
    def save_split_data(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        output_dir: str
    ):
        """保存分割后的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练数据
        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 保存验证数据
        with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # 保存测试数据
        with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分割数据已保存到: {output_dir}")


def create_data_loaders(
    train_dataset: NewsDataset,
    val_dataset: NewsDataset,
    test_dataset: NewsDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否使用内存固定
        
    Returns:
        训练、验证、测试数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试代码
    
    # 创建模拟数据
    sample_data = [
        {
            "title": "政府发布新政策",
            "content": "政府今天发布了一项新的经济政策，旨在促进经济发展。",
            "label": "政治/政府"
        },
        {
            "title": "科技公司股价上涨",
            "content": "多家科技公司股价今日大幅上涨，投资者信心增强。",
            "label": "商业/经济"
        },
        {
            "title": "人工智能新突破",
            "content": "研究人员在人工智能领域取得了重要突破，开发出新的算法。",
            "label": "科技/科学"
        }
    ]
    
    # 创建配置
    config = {
        "preprocessing": {
            "text_cleaning": {
                "remove_html": True,
                "normalize_whitespace": True,
                "min_length": 10,
                "max_length": 1000
            },
            "tokenization": {
                "method": "whitespace",
                "lowercase": True,
                "remove_stopwords": False
            },
            "vocabulary": {
                "min_freq": 1,
                "max_vocab_size": 1000
            }
        }
    }
    
    # 测试预处理器
    print("测试文本预处理器...")
    preprocessor = TextPreprocessor(config)
    
    # 构建词汇表
    texts = [f"{item['title']} {item['content']}" for item in sample_data]
    vocab_size = preprocessor.build_vocabulary(texts)
    print(f"词汇表大小: {vocab_size}")
    
    # 测试数据集
    print("\n测试数据集...")
    dataset = NewsDataset(sample_data, preprocessor)
    print(f"数据集大小: {len(dataset)}")
    print(f"类别分布: {dataset.get_class_distribution()}")
    print(f"统计信息: {dataset.get_statistics()}")
    
    # 测试数据加载器
    print("\n测试数据加载器...")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"批次 {batch_idx}:")
        print(f"  文本shape: {batch['text'].shape}")
        print(f"  标签: {batch['label']}")
        if batch_idx >= 1:  # 只看前两个批次
            break
    
    print("测试完成!")
