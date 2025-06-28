#!/usr/bin/env python3
"""
新闻文本分类训练脚本

支持完整的训练流程，包括数据预处理、模型训练、评估和保存。
"""

import sys
import os
import argparse
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from models.optimized_mlp import OptimizedTextMLP, create_model_from_config
from training.trainer import NewsClassificationTrainer
from preprocessing.llm_processor import LLMProcessor
from utils.data_loader import NewsDataset, NewsDataLoader, TextPreprocessor
from utils.logger import setup_logger
from evaluation.evaluator import ModelEvaluator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="新闻文本分类训练脚本")
    
    # 基本参数
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml", 
        help="模型配置文件路径"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default="config/llm_config.yaml",
        help="LLM配置文件路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="数据目录路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="输出目录路径"
    )
    
    # 数据相关参数
    parser.add_argument(
        "--data-file",
        type=str,
        help="指定数据文件路径"
    )
    parser.add_argument(
        "--use-llm-preprocessing",
        action="store_true",
        help="是否使用LLM预处理"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="限制数据样本数量（用于快速测试）"
    )
    
    # 训练相关参数
    parser.add_argument(
        "--epochs",
        type=int,
        help="训练轮数（覆盖配置文件）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="学习率（覆盖配置文件）"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="计算设备"
    )
    
    # 模式相关参数
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="运行完整流程（包括数据预处理）"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="快速测试模式"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="仅评估模式"
    )
    
    # 其他参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )
    
    return parser.parse_args()


def load_configs(args):
    """加载配置文件"""
    configs = {}
    
    # 加载训练配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            configs['training'] = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"训练配置文件不存在: {args.config}")
    
    # 加载模型配置
    if os.path.exists(args.model_config):
        with open(args.model_config, 'r', encoding='utf-8') as f:
            configs['model'] = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"模型配置文件不存在: {args.model_config}")
    
    # 加载LLM配置（如果需要）
    if args.use_llm_preprocessing and os.path.exists(args.llm_config):
        with open(args.llm_config, 'r', encoding='utf-8') as f:
            configs['llm'] = yaml.safe_load(f)
    
    return configs


def setup_environment(args):
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger("main", run_dir / "training.log", level=log_level)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    return device, run_dir, logger


def load_and_preprocess_data(args, configs, logger):
    """加载和预处理数据"""
    logger.info("开始加载和预处理数据...")
    
    # 创建数据加载器
    data_loader = NewsDataLoader(
        data_dir=args.data_dir,
        config=configs['model']
    )
    
    # 加载原始数据
    if args.data_file:
        raw_data = data_loader.load_from_file(args.data_file)
    else:
        raw_data = data_loader.load_from_directory(args.data_dir)
    
    logger.info(f"加载原始数据: {len(raw_data)} 个样本")
    
    # 限制样本数量（用于测试）
    if args.sample_size:
        raw_data = raw_data[:args.sample_size]
        logger.info(f"限制样本数量为: {len(raw_data)}")
    
    # LLM预处理
    processed_data = raw_data
    if args.use_llm_preprocessing:
        logger.info("开始LLM预处理...")
        llm_processor = LLMProcessor(config=configs.get('llm'))
        
        # 准备文章数据
        articles = []
        for item in raw_data:
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("content", "")
            })
        
        try:
            # 批量处理
            import asyncio
            results, stats = asyncio.run(llm_processor.process_batch(articles))
            logger.info(f"LLM预处理完成: 成功 {stats.successful_samples}/{stats.total_samples}")
            logger.info(f"总成本: ${stats.total_cost:.4f}")
            
            # 更新数据
            for i, (original, result) in enumerate(zip(raw_data, results)):
                if "clean" in result and result["clean"].success:
                    processed_data[i]["content"] = result["clean"].processed_text
                if "label" in result and result["label"].success:
                    # 如果LLM提供了高置信度标签，可以用于半监督学习
                    if result["label"].confidence > 0.8:
                        processed_data[i]["llm_label"] = result["label"].category
                        processed_data[i]["llm_confidence"] = result["label"].confidence
            
        except Exception as e:
            logger.error(f"LLM预处理失败: {e}")
            logger.info("继续使用原始数据...")
        
        finally:
            await llm_processor.close()
    
    # 文本预处理
    logger.info("开始文本预处理...")
    preprocessor = TextPreprocessor(config=configs['model'])
    
    # 构建词汇表
    texts = [item["content"] for item in processed_data]
    vocab_size = preprocessor.build_vocabulary(texts)
    logger.info(f"构建词汇表: {vocab_size} 个词汇")
    
    # 更新模型配置中的词汇表大小
    configs['model']['model']['embedding']['vocab_size'] = vocab_size
    
    # 创建数据集
    dataset = NewsDataset(
        processed_data,
        preprocessor,
        class_names=configs['model']['classes']['class_names']
    )
    
    logger.info(f"创建数据集: {len(dataset)} 个样本")
    
    return dataset, preprocessor, vocab_size


def split_dataset(dataset, configs, logger):
    """分割数据集"""
    logger.info("分割数据集...")
    
    # 获取分割比例
    data_config = configs['training'].get('data_loading', {})
    train_ratio = data_config.get('train_split', 0.7)
    val_ratio = data_config.get('val_split', 0.15)
    test_ratio = data_config.get('test_split', 0.15)
    
    # 计算样本数量
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"数据集分割: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}, 测试 {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, configs, logger):
    """创建数据加载器"""
    logger.info("创建数据加载器...")
    
    data_config = configs['training'].get('data_loading', {})
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据加载器创建完成: batch_size={batch_size}")
    
    return train_loader, val_loader, test_loader


def create_model(configs, device, logger):
    """创建模型"""
    logger.info("创建模型...")
    
    # 从配置创建模型
    model = create_model_from_config(configs['model'])
    
    # 打印模型信息
    model_info = model.get_model_info()
    logger.info(f"模型信息:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # 移动到设备
    model.to(device)
    
    return model


def train_model(model, train_loader, val_loader, configs, device, run_dir, logger, args):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 应用命令行参数覆盖
    training_config = configs['training'].copy()
    if args.epochs:
        training_config['training']['epochs'] = args.epochs
    if args.batch_size:
        training_config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        training_config['training']['learning_rate'] = args.learning_rate
    
    # 创建训练器
    trainer = NewsClassificationTrainer(
        model=model,
        config=training_config,
        device=device,
        log_dir=run_dir
    )
    
    # 从检查点恢复（如果指定）
    start_epoch = 0
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint_info = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint_info["epoch"]
        logger.info(f"从第 {start_epoch} 轮恢复训练")
    
    # 执行训练
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['training']['epochs'] - start_epoch
    )
    
    # 保存训练结果
    results_file = run_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    logger.info(f"训练完成!")
    logger.info(f"最佳F1分数: {training_results['best_metric']:.4f}")
    logger.info(f"训练时间: {training_results['total_time']:.1f}秒")
    
    return trainer, training_results


def evaluate_model(model, test_loader, configs, run_dir, logger):
    """评估模型"""
    logger.info("开始评估模型...")
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model=model,
        class_names=configs['model']['classes']['class_names']
    )
    
    # 执行评估
    eval_results = evaluator.evaluate(test_loader, return_detailed=True)
    
    # 分析预测结果
    analysis_results = evaluator.analyze_predictions(
        test_loader, 
        save_dir=run_dir / "evaluation"
    )
    
    # 保存评估结果
    eval_file = run_dir / "evaluation_results.json"
    with open(eval_file, 'w') as f:
        json.dump({
            "evaluation": eval_results,
            "analysis": analysis_results
        }, f, indent=2, default=str)
    
    # 打印主要指标
    logger.info("评估结果:")
    logger.info(f"  准确率: {eval_results['accuracy']:.4f}")
    logger.info(f"  宏平均F1: {eval_results['f1_macro']:.4f}")
    logger.info(f"  微平均F1: {eval_results['f1_micro']:.4f}")
    logger.info(f"  Cohen's Kappa: {eval_results['cohen_kappa']:.4f}")
    
    # 每类别结果
    for i, class_name in enumerate(configs['model']['classes']['class_names']):
        logger.info(f"  {class_name}:")
        logger.info(f"    精确率: {eval_results['precision_per_class'][i]:.4f}")
        logger.info(f"    召回率: {eval_results['recall_per_class'][i]:.4f}")
        logger.info(f"    F1分数: {eval_results['f1_per_class'][i]:.4f}")
    
    return eval_results, analysis_results


def save_final_model(model, preprocessor, configs, eval_results, run_dir, logger):
    """保存最终模型"""
    logger.info("保存最终模型...")
    
    # 创建模型包
    model_package = {
        "model_state_dict": model.state_dict(),
        "model_config": configs['model'],
        "preprocessor": preprocessor.get_config(),
        "evaluation_results": eval_results,
        "class_names": configs['model']['classes']['class_names'],
        "vocab_size": configs['model']['model']['embedding']['vocab_size'],
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存模型
    model_file = run_dir / "final_model.pt"
    torch.save(model_package, model_file)
    
    # 保存为ONNX格式（用于部署）
    try:
        dummy_input = torch.randint(0, 1000, (1, 50)).to(model.device)
        onnx_file = run_dir / "final_model.onnx"
        torch.onnx.export(
            model, dummy_input, onnx_file,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'},
                         'output': {0: 'batch_size'}}
        )
        logger.info(f"ONNX模型已保存: {onnx_file}")
    except Exception as e:
        logger.warning(f"ONNX导出失败: {e}")
    
    logger.info(f"最终模型已保存: {model_file}")
    
    return model_file


def generate_summary_report(training_results, eval_results, run_dir, logger):
    """生成汇总报告"""
    logger.info("生成汇总报告...")
    
    report = {
        "experiment_summary": {
            "timestamp": datetime.now().isoformat(),
            "run_directory": str(run_dir),
            "total_training_time": training_results.get("total_time", 0),
            "epochs_trained": training_results.get("epochs_trained", 0),
            "best_validation_metric": training_results.get("best_metric", 0)
        },
        "final_test_results": {
            "accuracy": eval_results["accuracy"],
            "f1_macro": eval_results["f1_macro"],
            "f1_micro": eval_results["f1_micro"],
            "precision_macro": eval_results["precision_macro"],
            "recall_macro": eval_results["recall_macro"],
            "cohen_kappa": eval_results["cohen_kappa"]
        },
        "per_class_results": {}
    }
    
    # 添加每类别结果
    class_names = ["政治/政府", "商业/经济", "科技/科学"]
    for i, class_name in enumerate(class_names):
        report["per_class_results"][class_name] = {
            "precision": eval_results["precision_per_class"][i],
            "recall": eval_results["recall_per_class"][i],
            "f1_score": eval_results["f1_per_class"][i],
            "support": eval_results["support_per_class"][i]
        }
    
    # 保存报告
    report_file = run_dir / "experiment_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 生成简要文本报告
    text_report = f"""
新闻文本分类实验报告
====================

实验时间: {report['experiment_summary']['timestamp']}
运行目录: {report['experiment_summary']['run_directory']}

训练结果:
- 训练时间: {report['experiment_summary']['total_training_time']:.1f}秒
- 训练轮数: {report['experiment_summary']['epochs_trained']}
- 最佳验证指标: {report['experiment_summary']['best_validation_metric']:.4f}

测试结果:
- 准确率: {report['final_test_results']['accuracy']:.4f}
- 宏平均F1: {report['final_test_results']['f1_macro']:.4f}
- 微平均F1: {report['final_test_results']['f1_micro']:.4f}
- Cohen's Kappa: {report['final_test_results']['cohen_kappa']:.4f}

各类别表现:
"""
    
    for class_name, metrics in report["per_class_results"].items():
        text_report += f"""
{class_name}:
  精确率: {metrics['precision']:.4f}
  召回率: {metrics['recall']:.4f}
  F1分数: {metrics['f1_score']:.4f}
  样本数: {metrics['support']}
"""
    
    # 保存文本报告
    text_report_file = run_dir / "experiment_summary.txt"
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    logger.info(f"汇总报告已保存: {report_file}")
    print("\n" + "="*50)
    print("实验完成!")
    print(f"结果目录: {run_dir}")
    print(f"准确率: {eval_results['accuracy']:.4f}")
    print(f"F1分数: {eval_results['f1_macro']:.4f}")
    print("="*50)
    
    return report


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置环境
    device, run_dir, logger = setup_environment(args)
    
    try:
        logger.info("="*50)
        logger.info("开始新闻文本分类训练")
        logger.info("="*50)
        logger.info(f"运行目录: {run_dir}")
        logger.info(f"计算设备: {device}")
        
        # 加载配置
        configs = load_configs(args)
        logger.info("配置文件加载成功")
        
        # 保存配置到运行目录
        config_save_path = run_dir / "configs.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(configs, f, default_flow_style=False)
        
        # 加载和预处理数据
        if not args.evaluate_only:
            dataset, preprocessor, vocab_size = load_and_preprocess_data(args, configs, logger)
            
            # 分割数据集
            train_dataset, val_dataset, test_dataset = split_dataset(dataset, configs, logger)
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = create_data_loaders(
                train_dataset, val_dataset, test_dataset, configs, logger
            )
        
        # 创建模型
        model = create_model(configs, device, logger)
        
        # 训练或评估
        if args.evaluate_only:
            # 仅评估模式 - 需要加载预训练模型
            if not args.resume:
                raise ValueError("评估模式需要指定 --resume 参数")
            
            # TODO: 加载测试数据
            logger.info("评估模式: 加载预训练模型")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # 训练模式
            trainer, training_results = train_model(
                model, train_loader, val_loader, configs, device, run_dir, logger, args
            )
        
        # 评估模型
        if not args.evaluate_only:
            eval_results, analysis_results = evaluate_model(
                model, test_loader, configs, run_dir, logger
            )
            
            # 保存最终模型
            model_file = save_final_model(
                model, preprocessor, configs, eval_results, run_dir, logger
            )
            
            # 生成汇总报告
            summary_report = generate_summary_report(
                training_results, eval_results, run_dir, logger
            )
        
        logger.info("训练脚本执行完成!")
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
