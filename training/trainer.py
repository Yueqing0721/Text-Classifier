"""
新闻文本分类训练器

支持混合精度训练、早停、学习率调度等现代训练技术，
专门针对小数据集优化。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb

from ..models.optimized_mlp import OptimizedTextMLP, LabelSmoothingCrossEntropy
from ..utils.logger import setup_logger
from ..utils.visualization import plot_training_curves, plot_confusion_matrix


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善阈值
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否更好"""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
    
    def update(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor):
        """更新指标"""
        # 转换为numpy数组
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # 计算指标
        accuracy = accuracy_score(target_np, pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_np, pred_np, average='macro', zero_division=0
        )
        
        # 记录指标
        self.metrics["loss"].append(loss)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["precision"].append(precision)
        self.metrics["recall"].append(recall)
        self.metrics["f1"].append(f1)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        return {
            metric: np.mean(values) if values else 0.0
            for metric, values in self.metrics.items()
        }


class NewsClassificationTrainer:
    """新闻文本分类训练器"""
    
    def __init__(
        self, 
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        log_dir: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
            device: 计算设备
            log_dir: 日志目录
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将模型移到设备上
        self.model.to(self.device)
        
        # 混合精度训练
        self.use_mixed_precision = config.get("training", {}).get("mixed_precision", {}).get("enabled", True)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
                logging.info("模型编译成功，将获得性能提升")
            except Exception as e:
                logging.warning(f"模型编译失败: {e}")
        
        # 设置日志
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("trainer", self.log_dir / "training.log")
        
        # TensorBoard
        if config.get("logging", {}).get("tensorboard", {}).get("enabled", True):
            self.writer = SummaryWriter(self.log_dir / "tensorboard")
        else:
            self.writer = None
        
        # Weights & Biases
        if config.get("logging", {}).get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config.get("logging", {}).get("wandb", {}).get("project", "news-classification"),
                config=config,
                dir=str(self.log_dir)
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # 初始化组件
        self._setup_training_components()
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": []
        }
        
        # 最佳模型跟踪
        self.best_model_state = None
        self.best_metric_value = 0.0
    
    def _setup_training_components(self):
        """设置训练组件"""
        training_config = self.config.get("training", {})
        
        # 优化器
        optimizer_config = training_config.get("optimizer", {})
        optimizer_name = optimizer_config.get("name", "adamw").lower()
        
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config.get("learning_rate", 0.001),
                weight_decay=training_config.get("weight_decay", 0.01),
                betas=optimizer_config.get("betas", [0.9, 0.999]),
                eps=optimizer_config.get("eps", 1e-8)
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.get("learning_rate", 0.001),
                betas=optimizer_config.get("betas", [0.9, 0.999]),
                eps=optimizer_config.get("eps", 1e-8)
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config.get("learning_rate", 0.001),
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=training_config.get("weight_decay", 0.01)
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 学习率调度器
        scheduler_config = training_config.get("scheduler", {})
        scheduler_name = scheduler_config.get("name", "onecycle").lower()
        
        if scheduler_name == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get("max_lr", 0.003),
                epochs=training_config.get("epochs", 30),
                steps_per_epoch=1,  # 会在训练时更新
                pct_start=scheduler_config.get("pct_start", 0.3),
                anneal_strategy=scheduler_config.get("anneal_strategy", "cos")
            )
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get("epochs", 30),
                eta_min=scheduler_config.get("eta_min", 1e-6)
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 3),
                min_lr=scheduler_config.get("min_lr", 1e-6)
            )
        else:
            self.scheduler = None
        
        # 损失函数
        loss_config = training_config.get("loss", {})
        if loss_config.get("name", "cross_entropy") == "cross_entropy":
            if loss_config.get("label_smoothing", 0) > 0:
                self.criterion = LabelSmoothingCrossEntropy(
                    smoothing=loss_config.get("label_smoothing", 0.1)
                )
            else:
                self.criterion = nn.CrossEntropyLoss(
                    weight=loss_config.get("class_weights")
                )
        else:
            raise ValueError(f"不支持的损失函数: {loss_config.get('name')}")
        
        # 早停
        early_stopping_config = training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get("patience", 5),
                min_delta=early_stopping_config.get("min_delta", 0.001),
                mode=early_stopping_config.get("mode", "max")
            )
        else:
            self.early_stopping = None
        
        # 梯度裁剪
        self.gradient_clipping = training_config.get("gradient_clipping", {})
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据移到设备
            inputs = batch["text"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, attention_mask)
                    loss = self.criterion(outputs, targets)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.gradient_clipping.get("enabled", True):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping.get("max_norm", 1.0)
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs, attention_mask)
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.gradient_clipping.get("enabled", True):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping.get("max_norm", 1.0)
                    )
                
                # 更新参数
                self.optimizer.step()
            
            # 更新学习率（对于OneCycleLR）
            if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # 计算预测
            predictions = torch.argmax(outputs, dim=1)
            
            # 更新指标
            metrics_tracker.update(loss.item(), predictions, targets)
            
            # 打印进度
            if batch_idx % self.config.get("logging", {}).get("console", {}).get("print_frequency", 10) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Batch {batch_idx}/{total_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.6f}"
                )
        
        return metrics_tracker.get_average_metrics()
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["text"].to(self.device, non_blocking=True)
                targets = batch["label"].to(self.device, non_blocking=True)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs, attention_mask)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs, attention_mask)
                    loss = self.criterion(outputs, targets)
                
                predictions = torch.argmax(outputs, dim=1)
                
                # 收集预测结果用于详细分析
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 更新指标
                metrics_tracker.update(loss.item(), predictions, targets)
        
        # 计算详细指标
        metrics = metrics_tracker.get_average_metrics()
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # 计算每类别的精确率、召回率、F1
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average=None, zero_division=0
        )
        
        metrics["precision_per_class"] = precision_per_class.tolist()
        metrics["recall_per_class"] = recall_per_class.tolist()
        metrics["f1_per_class"] = f1_per_class.tolist()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            
        Returns:
            训练历史和结果
        """
        epochs = epochs or self.config.get("training", {}).get("epochs", 30)
        
        # 更新OneCycleLR的steps_per_epoch
        if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
            self.scheduler.total_steps = epochs * len(train_loader)
        
        self.logger.info(f"开始训练，共 {epochs} 个epoch")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"可训练参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # 更新学习率（除了OneCycleLR）
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("f1", train_metrics["f1"]))
                else:
                    self.scheduler.step()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)
            
            if val_metrics:
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_f1"].append(val_metrics["f1"])
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("F1/Train", train_metrics["f1"], epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                if val_metrics:
                    self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
                    self.writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
                    self.writer.add_scalar("F1/Val", val_metrics["f1"], epoch)
            
            # 记录到Weights & Biases
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1": train_metrics["f1"],
                    "learning_rate": current_lr
                }
                if val_metrics:
                    log_dict.update({
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/f1": val_metrics["f1"]
                    })
                wandb.log(log_dict)
            
            # 保存最佳模型
            current_metric = val_metrics.get("f1", train_metrics["f1"])
            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, is_best=True)
            
            # 早停检查
            if self.early_stopping:
                if self.early_stopping(current_metric):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            log_msg = (
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Train F1: {train_metrics['f1']:.4f}"
            )
            
            if val_metrics:
                log_msg += (
                    f", Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            
            log_msg += f", Time: {epoch_time:.1f}s, LR: {current_lr:.6f}"
            self.logger.info(log_msg)
        
        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time:.1f}秒")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"已加载最佳模型 (F1: {self.best_metric_value:.4f})")
        
        # 保存训练历史
        self.save_training_history()
        
        # 生成训练曲线图
        self.plot_training_curves()
        
        return {
            "history": self.history,
            "best_metric": self.best_metric_value,
            "total_time": total_time,
            "epochs_trained": len(self.history["train_loss"])
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint_dir = Path(self.config.get("checkpointing", {}).get("save_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric_value": self.best_metric_value,
            "history": self.history,
            "config": self.config
        }
        
        if is_best:
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存最佳模型: {checkpoint_path}")
        
        # 定期保存
        save_frequency = self.config.get("checkpointing", {}).get("save_frequency", 5)
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
    
    def save_training_history(self):
        """保存训练历史"""
        history_file = self.log_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        try:
            fig = plot_training_curves(self.history)
            fig.savefig(self.log_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_figure("Training_Curves", fig)
            
        except Exception as e:
            self.logger.warning(f"绘制训练曲线失败: {e}")
    
    def close(self):
        """关闭训练器"""
        if self.writer:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None, scheduler = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        
    Returns:
        检查点信息
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric_value": checkpoint.get("best_metric_value", 0.0),
        "history": checkpoint.get("history", {}),
        "config": checkpoint.get("config", {})
    }


if __name__ == "__main__":
    # 测试训练器
    from ..models.optimized_mlp import OptimizedTextMLP
    
    # 创建模型
    model = OptimizedTextMLP(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dims=[256, 128],
        num_classes=3
    )
    
    # 测试配置
    config = {
        "training": {
            "epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "batch_size": 32,
            "mixed_precision": {"enabled": True}
        },
        "logging": {
            "tensorboard": {"enabled": False},
            "wandb": {"enabled": False}
        }
    }
    
    # 创建训练器
    trainer = NewsClassificationTrainer(model, config)
    
    print("训练器创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
