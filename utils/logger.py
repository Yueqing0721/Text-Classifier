"""
日志工具模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import colorlog


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None,
    color_output: bool = True
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        format_string: 自定义格式字符串
        color_output: 是否使用彩色输出
        
    Returns:
        配置好的日志器
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if color_output:
            # 彩色控制台输出
            color_format = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            console_formatter = colorlog.ColoredFormatter(
                color_format,
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_formatter = logging.Formatter(
                format_string,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output and log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """训练日志器"""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = None,
        level: int = logging.INFO
    ):
        """
        初始化训练日志器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        
        # 创建主日志器
        self.logger = setup_logger(
            name="training",
            log_file=self.log_dir / f"{experiment_name}.log",
            level=level
        )
        
        # 创建不同类型的日志器
        self.training_logger = setup_logger(
            name="training.train",
            log_file=self.log_dir / f"{experiment_name}_training.log",
            level=level,
            console_output=False
        )
        
        self.evaluation_logger = setup_logger(
            name="training.eval",
            log_file=self.log_dir / f"{experiment_name}_evaluation.log",
            level=level,
            console_output=False
        )
        
        self.error_logger = setup_logger(
            name="training.error",
            log_file=self.log_dir / f"{experiment_name}_errors.log",
            level=logging.ERROR,
            console_output=False
        )
        
        # 记录实验开始
        self.logger.info(f"开始实验: {experiment_name}")
        self.logger.info(f"日志目录: {self.log_dir}")
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        **kwargs
    ):
        """记录训练步骤"""
        message = (
            f"Epoch {epoch}, Step {step}: "
            f"Loss={loss:.4f}, Acc={accuracy:.4f}, LR={learning_rate:.6f}"
        )
        
        for key, value in kwargs.items():
            message += f", {key}={value}"
        
        self.training_logger.info(message)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float = None,
        val_acc: float = None,
        epoch_time: float = None,
        **kwargs
    ):
        """记录epoch总结"""
        message = (
            f"Epoch {epoch} Summary: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
        )
        
        if val_loss is not None:
            message += f", Val Loss={val_loss:.4f}"
        if val_acc is not None:
            message += f", Val Acc={val_acc:.4f}"
        if epoch_time is not None:
            message += f", Time={epoch_time:.1f}s"
        
        for key, value in kwargs.items():
            message += f", {key}={value}"
        
        self.logger.info(message)
    
    def log_evaluation_results(self, metrics: dict, dataset_name: str = "test"):
        """记录评估结果"""
        self.evaluation_logger.info(f"=== {dataset_name.upper()} EVALUATION RESULTS ===")
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.evaluation_logger.info(f"{metric_name}: {value:.4f}")
            else:
                self.evaluation_logger.info(f"{metric_name}: {value}")
        
        self.logger.info(f"{dataset_name}评估完成，详见评估日志")
    
    def log_hyperparameters(self, params: dict):
        """记录超参数"""
        self.logger.info("=== HYPERPARAMETERS ===")
        for param_name, value in params.items():
            self.logger.info(f"{param_name}: {value}")
    
    def log_model_info(self, model_info: dict):
        """记录模型信息"""
        self.logger.info("=== MODEL INFORMATION ===")
        for info_name, value in model_info.items():
            self.logger.info(f"{info_name}: {value}")
    
    def log_dataset_info(self, dataset_info: dict):
        """记录数据集信息"""
        self.logger.info("=== DATASET INFORMATION ===")
        for info_name, value in dataset_info.items():
            self.logger.info(f"{info_name}: {value}")
    
    def log_error(self, error_message: str, exception: Exception = None):
        """记录错误"""
        self.error_logger.error(error_message)
        if exception:
            self.error_logger.exception(exception)
        
        # 也记录到主日志
        self.logger.error(error_message)
    
    def log_experiment_completion(self, final_metrics: dict, total_time: float):
        """记录实验完成"""
        self.logger.info("=== EXPERIMENT COMPLETED ===")
        self.logger.info(f"Total training time: {total_time:.1f} seconds")
        
        self.logger.info("Final metrics:")
        for metric_name, value in final_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")


class MetricsLogger:
    """指标日志器"""
    
    def __init__(self, log_file: Union[str, Path]):
        """
        初始化指标日志器
        
        Args:
            log_file: 指标日志文件
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入CSV头部
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("timestamp,epoch,step,metric_name,metric_value,dataset,notes\n")
    
    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        epoch: int = None,
        step: int = None,
        dataset: str = "train",
        notes: str = ""
    ):
        """记录单个指标"""
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{epoch},{step},{metric_name},{metric_value},{dataset},{notes}\n")
    
    def log_metrics_dict(
        self,
        metrics: dict,
        epoch: int = None,
        step: int = None,
        dataset: str = "train",
        notes: str = ""
    ):
        """记录指标字典"""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.log_metric(metric_name, metric_value, epoch, step, dataset, notes)


def configure_logging_for_library(library_name: str, level: int = logging.WARNING):
    """
    配置第三方库的日志级别
    
    Args:
        library_name: 库名称
        level: 日志级别
    """
    library_logger = logging.getLogger(library_name)
    library_logger.setLevel(level)


def setup_distributed_logging(rank: int, world_size: int):
    """
    设置分布式训练的日志配置
    
    Args:
        rank: 进程排名
        world_size: 总进程数
    """
    # 只在主进程显示详细日志
    if rank == 0:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 为每个进程创建单独的日志文件
    log_file = f"logs/distributed_rank_{rank}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        f'[Rank {rank}/{world_size}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


class ProgressLogger:
    """进度日志器"""
    
    def __init__(self, total_steps: int, log_frequency: int = 10):
        """
        初始化进度日志器
        
        Args:
            total_steps: 总步数
            log_frequency: 日志频率
        """
        self.total_steps = total_steps
        self.log_frequency = log_frequency
        self.current_step = 0
        self.start_time = datetime.now()
        
        self.logger = logging.getLogger("progress")
    
    def update(self, step: int = None, **metrics):
        """更新进度"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if self.current_step % self.log_frequency == 0 or self.current_step == self.total_steps:
            progress = self.current_step / self.total_steps * 100
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            if self.current_step > 0:
                eta = elapsed_time / self.current_step * (self.total_steps - self.current_step)
                eta_str = f"ETA: {eta:.0f}s"
            else:
                eta_str = "ETA: N/A"
            
            message = f"Progress: {self.current_step}/{self.total_steps} ({progress:.1f}%) - {eta_str}"
            
            if metrics:
                metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                      for k, v in metrics.items()])
                message += f" - {metric_str}"
            
            self.logger.info(message)


# 设置一些常用库的日志级别
configure_logging_for_library("urllib3", logging.WARNING)
configure_logging_for_library("requests", logging.WARNING)
configure_logging_for_library("matplotlib", logging.WARNING)
configure_logging_for_library("PIL", logging.WARNING)


if __name__ == "__main__":
    # 测试日志功能
    
    # 测试基本日志器
    print("测试基本日志器...")
    logger = setup_logger("test", "logs/test.log")
    logger.debug("这是调试信息")
    logger.info("这是信息")
    logger.warning("这是警告")
    logger.error("这是错误")
    
    # 测试训练日志器
    print("\n测试训练日志器...")
    training_logger = TrainingLogger("logs/training", "test_experiment")
    
    # 记录超参数
    training_logger.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # 记录训练步骤
    for epoch in range(3):
        for step in range(5):
            training_logger.log_training_step(
                epoch=epoch,
                step=step,
                loss=1.0 - step * 0.1,
                accuracy=0.5 + step * 0.1,
                learning_rate=0.001
            )
        
        training_logger.log_epoch_summary(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.2,
            train_acc=0.6 + epoch * 0.1,
            val_loss=1.1 - epoch * 0.2,
            val_acc=0.55 + epoch * 0.1,
            epoch_time=10.5
        )
    
    # 记录最终结果
    training_logger.log_experiment_completion(
        final_metrics={"accuracy": 0.85, "f1_score": 0.83},
        total_time=150.0
    )
    
    # 测试指标日志器
    print("\n测试指标日志器...")
    metrics_logger = MetricsLogger("logs/metrics.csv")
    
    for epoch in range(3):
        metrics_logger.log_metrics_dict({
            "train_loss": 1.0 - epoch * 0.2,
            "train_acc": 0.6 + epoch * 0.1,
            "val_loss": 1.1 - epoch * 0.2,
            "val_acc": 0.55 + epoch * 0.1
        }, epoch=epoch, dataset="validation")
    
    # 测试进度日志器
    print("\n测试进度日志器...")
    progress_logger = ProgressLogger(total_steps=100, log_frequency=25)
    
    for i in range(1, 101):
        progress_logger.update(loss=1.0/i, accuracy=i/100)
        if i % 25 == 0:
            import time
            time.sleep(0.1)  # 模拟处理时间
    
    print("日志测试完成!")
