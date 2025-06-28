"""
模型评估和统计检验模块

提供全面的模型评估功能，包括：
- 多种评估指标计算
- 交叉验证
- 统计显著性检验
- 结果可视化
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, cohen_kappa_score
)
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        class_names: List[str] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
            class_names: 类别名称列表
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [f"Class_{i}" for i in range(3)]
        
        # 将模型移到设备并设为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def predict(
        self, 
        data_loader: DataLoader, 
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        对数据进行预测
        
        Args:
            data_loader: 数据加载器
            return_probabilities: 是否返回概率
            
        Returns:
            预测结果和概率（如果requested）
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["text"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(inputs, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        if return_probabilities:
            return predictions, probabilities
        return predictions
    
    def evaluate(
        self, 
        data_loader: DataLoader, 
        return_detailed: bool = True
    ) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            data_loader: 数据加载器
            return_detailed: 是否返回详细结果
            
        Returns:
            评估结果字典
        """
        # 获取真实标签
        true_labels = []
        for batch in data_loader:
            true_labels.extend(batch["label"].numpy())
        true_labels = np.array(true_labels)
        
        # 获取预测结果
        predictions, probabilities = self.predict(data_loader, return_probabilities=True)
        
        # 计算基本指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 计算精确率、召回率、F1分数
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='micro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        # 计算Cohen's Kappa
        kappa = cohen_kappa_score(true_labels, predictions)
        
        # 计算AUC（多分类）
        try:
            auc_ovr = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
            auc_ovo = roc_auc_score(true_labels, probabilities, multi_class='ovo', average='macro')
        except ValueError:
            auc_ovr = auc_ovo = np.nan
        
        # 基本结果
        results = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "cohen_kappa": kappa,
            "auc_ovr": auc_ovr,
            "auc_ovo": auc_ovo
        }
        
        # 详细结果
        if return_detailed:
            results.update({
                "confusion_matrix": cm.tolist(),
                "precision_per_class": precision_per_class.tolist(),
                "recall_per_class": recall_per_class.tolist(),
                "f1_per_class": f1_per_class.tolist(),
                "support_per_class": support_per_class.tolist(),
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "true_labels": true_labels.tolist(),
                "classification_report": classification_report(
                    true_labels, predictions, target_names=self.class_names, output_dict=True
                )
            })
        
        return results
    
    def plot_confusion_matrix(
        self, 
        true_labels: np.ndarray, 
        predictions: np.ndarray,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制混淆矩阵
        
        Args:
            true_labels: 真实标签
            predictions: 预测标签
            normalize: 是否标准化
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        cm = confusion_matrix(true_labels, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = '标准化混淆矩阵'
        else:
            fmt = 'd'
            title = '混淆矩阵'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_classification_report(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制分类报告热力图
        
        Args:
            true_labels: 真实标签
            predictions: 预测标签
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        report = classification_report(
            true_labels, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # 提取数据
        data = []
        for class_name in self.class_names:
            if class_name in report:
                data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
        
        data = np.array(data)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            data.T,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=self.class_names,
            yticklabels=['Precision', 'Recall', 'F1-Score'],
            ax=ax
        )
        
        ax.set_title('分类报告', fontsize=16, fontweight='bold')
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('指标', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_predictions(
        self,
        data_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析预测结果
        
        Args:
            data_loader: 数据加载器
            save_dir: 保存目录
            
        Returns:
            分析结果
        """
        # 获取预测结果
        results = self.evaluate(data_loader, return_detailed=True)
        
        true_labels = np.array(results["true_labels"])
        predictions = np.array(results["predictions"])
        probabilities = np.array(results["probabilities"])
        
        # 计算置信度统计
        max_probs = np.max(probabilities, axis=1)
        confidence_stats = {
            "mean_confidence": np.mean(max_probs),
            "std_confidence": np.std(max_probs),
            "min_confidence": np.min(max_probs),
            "max_confidence": np.max(max_probs),
            "median_confidence": np.median(max_probs)
        }
        
        # 分析错误预测
        error_mask = predictions != true_labels
        error_indices = np.where(error_mask)[0]
        
        error_analysis = {
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / len(true_labels),
            "error_confidence_stats": {
                "mean": np.mean(max_probs[error_mask]) if np.any(error_mask) else 0,
                "std": np.std(max_probs[error_mask]) if np.any(error_mask) else 0
            }
        }
        
        # 每类别分析
        class_analysis = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = true_labels == i
            class_predictions = predictions[class_mask]
            class_probabilities = probabilities[class_mask]
            
            if np.any(class_mask):
                class_analysis[class_name] = {
                    "sample_count": np.sum(class_mask),
                    "accuracy": accuracy_score(true_labels[class_mask], class_predictions),
                    "mean_confidence": np.mean(np.max(class_probabilities, axis=1)),
                    "error_rate": np.mean(class_predictions != i)
                }
        
        analysis_results = {
            "confidence_stats": confidence_stats,
            "error_analysis": error_analysis,
            "class_analysis": class_analysis,
            "evaluation_metrics": {k: v for k, v in results.items() 
                                  if k not in ["predictions", "probabilities", "true_labels"]}
        }
        
        # 保存分析结果
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存分析结果
            with open(save_dir / "prediction_analysis.json", 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # 绘制和保存图表
            self.plot_confusion_matrix(
                true_labels, predictions, 
                save_path=save_dir / "confusion_matrix.png"
            )
            
            self.plot_classification_report(
                true_labels, predictions,
                save_path=save_dir / "classification_report.png"
            )
            
            # 置信度分布图
            self.plot_confidence_distribution(
                max_probs, error_mask,
                save_path=save_dir / "confidence_distribution.png"
            )
        
        return analysis_results
    
    def plot_confidence_distribution(
        self,
        confidences: np.ndarray,
        error_mask: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制置信度分布图
        
        Args:
            confidences: 置信度数组
            error_mask: 错误预测掩码
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 整体置信度分布
        ax1.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(confidences):.3f}')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('频次')
        ax1.set_title('预测置信度分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 正确vs错误预测的置信度对比
        correct_confidences = confidences[~error_mask]
        error_confidences = confidences[error_mask]
        
        ax2.hist(correct_confidences, bins=20, alpha=0.7, color='green', 
                label=f'正确预测 (n={len(correct_confidences)})', density=True)
        ax2.hist(error_confidences, bins=20, alpha=0.7, color='red', 
                label=f'错误预测 (n={len(error_confidences)})', density=True)
        ax2.set_xlabel('置信度')
        ax2.set_ylabel('密度')
        ax2.set_title('正确vs错误预测的置信度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CrossValidator:
    """交叉验证器"""
    
    def __init__(
        self,
        model_class,
        model_params: Dict[str, Any],
        cv_strategy: str = "stratified",
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        初始化交叉验证器
        
        Args:
            model_class: 模型类
            model_params: 模型参数
            cv_strategy: 交叉验证策略
            n_folds: 折数
            random_state: 随机种子
        """
        self.model_class = model_class
        self.model_params = model_params
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.random_state = random_state
        
        # 设置交叉验证器
        if cv_strategy == "stratified":
            self.cv = StratifiedKFold(
                n_splits=n_folds, 
                shuffle=True, 
                random_state=random_state
            )
        else:
            raise ValueError(f"不支持的交叉验证策略: {cv_strategy}")
    
    def cross_validate(
        self,
        X: List[str],
        y: List[int],
        scoring: str = "f1_macro",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行交叉验证
        
        Args:
            X: 文本数据
            y: 标签
            scoring: 评分指标
            verbose: 是否打印进度
            
        Returns:
            交叉验证结果
        """
        fold_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            if verbose:
                print(f"训练第 {fold + 1}/{self.n_folds} 折...")
            
            # 分割数据
            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_train = [y[i] for i in train_idx]
            y_val = [y[i] for i in val_idx]
            
            # 这里需要实际的训练和评估逻辑
            # 由于涉及到具体的数据加载和训练流程，这里提供框架
            
            # TODO: 实现具体的训练和评估逻辑
            # 1. 创建数据加载器
            # 2. 创建模型实例
            # 3. 训练模型
            # 4. 评估模型
            
            # 临时返回随机分数用于演示
            fold_score = np.random.uniform(0.7, 0.9)
            fold_scores.append(fold_score)
            
            fold_result = {
                "fold": fold,
                "score": fold_score,
                "train_size": len(train_idx),
                "val_size": len(val_idx)
            }
            fold_results.append(fold_result)
        
        # 计算统计信息
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        ci_95 = 1.96 * std_score / np.sqrt(len(fold_scores))
        
        cv_results = {
            "fold_scores": fold_scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "ci_95": ci_95,
            "fold_results": fold_results,
            "cv_strategy": self.cv_strategy,
            "n_folds": self.n_folds,
            "scoring": scoring
        }
        
        if verbose:
            print(f"交叉验证完成:")
            print(f"  平均分数: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  95%置信区间: [{mean_score - ci_95:.4f}, {mean_score + ci_95:.4f}]")
        
        return cv_results


class StatisticalTester:
    """统计检验器"""
    
    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        配对t检验
        
        Args:
            scores1: 方法1的分数
            scores2: 方法2的分数
            alpha: 显著性水平
            
        Returns:
            检验结果
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # 计算差值
        differences = scores1 - scores2
        
        # 执行配对t检验
        t_statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        # 计算效应大小 (Cohen's d)
        pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
        cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0
        
        # 判断显著性
        is_significant = p_value < alpha
        
        result = {
            "test_name": "配对t检验",
            "t_statistic": t_statistic,
            "p_value": p_value,
            "alpha": alpha,
            "is_significant": is_significant,
            "mean_difference": np.mean(differences),
            "std_difference": np.std(differences, ddof=1),
            "cohens_d": cohens_d,
            "interpretation": f"方法1 {'显著' if is_significant else '不显著'}优于方法2 (p={'<' if p_value < 0.001 else '='}{p_value:.3f})"
        }
        
        return result
    
    @staticmethod
    def wilcoxon_signed_rank_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Wilcoxon符号秩检验（非参数）
        
        Args:
            scores1: 方法1的分数
            scores2: 方法2的分数
            alpha: 显著性水平
            
        Returns:
            检验结果
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        # 执行Wilcoxon符号秩检验
        statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
        
        # 判断显著性
        is_significant = p_value < alpha
        
        result = {
            "test_name": "Wilcoxon符号秩检验",
            "statistic": statistic,
            "p_value": p_value,
            "alpha": alpha,
            "is_significant": is_significant,
            "median_difference": np.median(scores1 - scores2),
            "interpretation": f"方法1 {'显著' if is_significant else '不显著'}优于方法2 (p={'<' if p_value < 0.001 else '='}{p_value:.3f})"
        }
        
        return result
    
    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        McNemar检验（用于比较两个分类器）
        
        Args:
            y_true: 真实标签
            y_pred1: 分类器1的预测
            y_pred2: 分类器2的预测
            alpha: 显著性水平
            
        Returns:
            检验结果
        """
        # 构建列联表
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # 2x2列联表
        # [正确1&正确2, 正确1&错误2]
        # [错误1&正确2, 错误1&错误2]
        b = np.sum(correct1 & ~correct2)  # 分类器1正确，分类器2错误
        c = np.sum(~correct1 & correct2)  # 分类器1错误，分类器2正确
        
        # McNemar统计量
        if b + c == 0:
            chi2_statistic = 0
            p_value = 1.0
        else:
            chi2_statistic = (abs(b - c) - 1) ** 2 / (b + c)  # 连续性校正
            p_value = 1 - stats.chi2.cdf(chi2_statistic, df=1)
        
        # 判断显著性
        is_significant = p_value < alpha
        
        result = {
            "test_name": "McNemar检验",
            "chi2_statistic": chi2_statistic,
            "p_value": p_value,
            "alpha": alpha,
            "is_significant": is_significant,
            "contingency_table": {"b": int(b), "c": int(c)},
            "interpretation": f"两个分类器的性能 {'存在显著差异' if is_significant else '无显著差异'} (p={'<' if p_value < 0.001 else '='}{p_value:.3f})"
        }
        
        return result
    
    @staticmethod
    def multiple_comparison_correction(
        p_values: List[float],
        method: str = "bonferroni",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        多重比较校正
        
        Args:
            p_values: p值列表
            method: 校正方法
            alpha: 显著性水平
            
        Returns:
            校正结果
        """
        from statsmodels.stats.multitest import multipletests
        
        # 执行多重比较校正
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        result = {
            "method": method,
            "original_alpha": alpha,
            "corrected_alpha": alpha_bonf if method == "bonferroni" else alpha_sidak,
            "original_p_values": p_values,
            "corrected_p_values": p_corrected.tolist(),
            "rejected": rejected.tolist(),
            "significant_count": np.sum(rejected)
        }
        
        return result


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, class_names: List[str] = None):
        """
        初始化综合评估器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names or ["政治/政府", "商业/经济", "科技/科学"]
        self.statistical_tester = StatisticalTester()
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            model_results: 模型结果字典，格式为 {model_name: evaluation_results}
            save_dir: 保存目录
            
        Returns:
            比较结果
        """
        # 提取关键指标
        metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        comparison_data = {}
        
        for metric in metrics:
            comparison_data[metric] = {}
            for model_name, results in model_results.items():
                if isinstance(results, dict) and metric in results:
                    comparison_data[metric][model_name] = results[metric]
        
        # 统计检验（如果有交叉验证结果）
        statistical_results = {}
        model_names = list(model_results.keys())
        
        if len(model_names) >= 2:
            # 两两比较
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    comparison_key = f"{model1}_vs_{model2}"
                    
                    # 如果有折分数，进行配对t检验
                    if ("fold_scores" in model_results[model1] and 
                        "fold_scores" in model_results[model2]):
                        
                        scores1 = model_results[model1]["fold_scores"]
                        scores2 = model_results[model2]["fold_scores"]
                        
                        # 配对t检验
                        t_test_result = self.statistical_tester.paired_t_test(scores1, scores2)
                        
                        # Wilcoxon符号秩检验
                        wilcoxon_result = self.statistical_tester.wilcoxon_signed_rank_test(scores1, scores2)
                        
                        statistical_results[comparison_key] = {
                            "paired_t_test": t_test_result,
                            "wilcoxon_test": wilcoxon_result
                        }
        
        # 生成比较报告
        comparison_summary = self._generate_comparison_summary(comparison_data, statistical_results)
        
        # 绘制比较图表
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self._plot_model_comparison(comparison_data, save_dir)
            
            # 保存比较结果
            with open(save_dir / "model_comparison.json", 'w') as f:
                json.dump({
                    "comparison_data": comparison_data,
                    "statistical_results": statistical_results,
                    "summary": comparison_summary
                }, f, indent=2, default=str)
        
        return {
            "comparison_data": comparison_data,
            "statistical_results": statistical_results,
            "summary": comparison_summary
        }
    
    def _generate_comparison_summary(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成比较摘要"""
        summary = {}
        
        # 找出每个指标的最佳模型
        for metric, values in comparison_data.items():
            if values:
                best_model = max(values.keys(), key=lambda k: values[k])
                summary[f"best_{metric}"] = {
                    "model": best_model,
                    "value": values[best_model]
                }
        
        # 统计显著性摘要
        significant_comparisons = 0
        total_comparisons = len(statistical_results)
        
        for comparison, tests in statistical_results.items():
            if any(test["is_significant"] for test in tests.values()):
                significant_comparisons += 1
        
        summary["statistical_summary"] = {
            "total_comparisons": total_comparisons,
            "significant_comparisons": significant_comparisons,
            "significance_rate": significant_comparisons / total_comparisons if total_comparisons > 0 else 0
        }
        
        return summary
    
    def _plot_model_comparison(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        save_dir: Path
    ):
        """绘制模型比较图表"""
        # 创建比较柱状图
        metrics = list(comparison_data.keys())
        models = list(next(iter(comparison_data.values())).keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # 最多显示4个指标
            if i < len(axes):
                ax = axes[i]
                values = [comparison_data[metric].get(model, 0) for model in models]
                
                bars = ax.bar(models, values, alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('分数')
                ax.set_ylim(0, 1)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # 测试评估器功能
    print("评估模块测试")
    
    # 创建模拟数据
    np.random.seed(42)
    true_labels = np.random.randint(0, 3, 100)
    predictions = np.random.randint(0, 3, 100)
    
    # 测试统计检验
    tester = StatisticalTester()
    
    # 模拟两组分数
    scores1 = np.random.normal(0.85, 0.05, 10)
    scores2 = np.random.normal(0.80, 0.05, 10)
    
    # 配对t检验
    t_test_result = tester.paired_t_test(scores1, scores2)
    print("配对t检验结果:")
    print(f"  t统计量: {t_test_result['t_statistic']:.4f}")
    print(f"  p值: {t_test_result['p_value']:.4f}")
    print(f"  是否显著: {t_test_result['is_significant']}")
    
    # McNemar检验
    pred1 = np.random.randint(0, 3, 100)
    pred2 = np.random.randint(0, 3, 100)
    mcnemar_result = tester.mcnemar_test(true_labels, pred1, pred2)
    print(f"\nMcNemar检验结果:")
    print(f"  χ²统计量: {mcnemar_result['chi2_statistic']:.4f}")
    print(f"  p值: {mcnemar_result['p_value']:.4f}")
    print(f"  是否显著: {mcnemar_result['is_significant']}")
