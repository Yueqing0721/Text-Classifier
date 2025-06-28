"""
LLM辅助预处理核心模块

使用OpenAI GPT-4o-mini等大语言模型进行智能文本预处理，包括：
- 文本清洗和格式化
- 质量评估和筛选
- 自动标注和分类
- 批量处理和成本控制
"""

import asyncio
import aiohttp
import json
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import os
from datetime import datetime, timedelta
import pickle

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    success: bool
    original_text: str
    processed_text: Optional[str] = None
    quality_score: Optional[float] = None
    category: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None


@dataclass
class BatchProcessingStats:
    """批处理统计信息"""
    total_samples: int
    successful_samples: int
    failed_samples: int
    total_tokens_used: int
    total_cost: float
    processing_time: float
    average_quality_score: float


class CostTracker:
    """成本跟踪器"""
    
    def __init__(self, daily_limit: float = 10.0, monthly_limit: float = 200.0):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.cost_log = []
        
    def add_cost(self, cost: float, tokens: int, model: str = "gpt-4o-mini"):
        """添加成本记录"""
        record = {
            "timestamp": datetime.now(),
            "cost": cost,
            "tokens": tokens,
            "model": model
        }
        self.cost_log.append(record)
        
    def get_daily_cost(self) -> float:
        """获取今日成本"""
        today = datetime.now().date()
        daily_costs = [
            record["cost"] for record in self.cost_log
            if record["timestamp"].date() == today
        ]
        return sum(daily_costs)
    
    def get_monthly_cost(self) -> float:
        """获取本月成本"""
        current_month = datetime.now().replace(day=1)
        monthly_costs = [
            record["cost"] for record in self.cost_log
            if record["timestamp"] >= current_month
        ]
        return sum(monthly_costs)
    
    def check_limits(self) -> Dict[str, bool]:
        """检查成本限制"""
        return {
            "daily_ok": self.get_daily_cost() < self.daily_limit,
            "monthly_ok": self.get_monthly_cost() < self.monthly_limit
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/llm_results/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, text: str, task_type: str) -> str:
        """生成缓存键"""
        content = f"{task_type}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, task_type: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        cache_key = self._get_cache_key(text, task_type)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 检查缓存是否过期（30天）
                if datetime.now() - cached_data["timestamp"] < timedelta(days=30):
                    return cached_data["result"]
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def set(self, text: str, task_type: str, result: Dict[str, Any]):
        """设置缓存结果"""
        cache_key = self._get_cache_key(text, task_type)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        cached_data = {
            "timestamp": datetime.now(),
            "result": result
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logger.warning(f"写入缓存失败: {e}")


class LLMProcessor:
    """LLM预处理器主类"""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化LLM处理器
        
        Args:
            config_path: 配置文件路径
            config: 配置字典（优先级高于config_path）
        """
        # 加载配置
        if config:
            self.config = config
        elif config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 默认配置
            self.config = self._get_default_config()
        
        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or self.config.get("llm", {}).get("api_key"),
            base_url=self.config.get("llm", {}).get("api", {}).get("base_url")
        )
        
        # 初始化组件
        self.cost_tracker = CostTracker(
            daily_limit=self.config.get("cost_control", {}).get("max_daily_cost", 10.0),
            monthly_limit=self.config.get("cost_control", {}).get("max_monthly_cost", 200.0)
        )
        
        self.cache_manager = CacheManager(
            cache_dir=self.config.get("caching", {}).get("cache_dir", "cache/llm_results/")
        )
        
        # 速率限制信号量
        max_concurrent = self.config.get("llm", {}).get("rate_limiting", {}).get("concurrent_requests", 5)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "parameters": {
                    "temperature": 0.0,
                    "max_tokens": 2048
                },
                "rate_limiting": {
                    "requests_per_minute": 100,
                    "concurrent_requests": 5
                }
            },
            "preprocessing_tasks": {
                "text_cleaning": {"enabled": True},
                "quality_assessment": {"enabled": True, "quality_threshold": 70},
                "auto_labeling": {"enabled": True, "confidence_threshold": 0.8}
            },
            "cost_control": {
                "max_daily_cost": 10.0,
                "max_monthly_cost": 200.0
            },
            "caching": {"enabled": True, "cache_dir": "cache/llm_results/"}
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _make_api_request(
        self, 
        messages: List[Dict[str, str]], 
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        发送API请求（带重试机制）
        
        Args:
            messages: 消息列表
            task_type: 任务类型
            
        Returns:
            API响应结果
        """
        async with self.semaphore:
            try:
                # 检查成本限制
                limits = self.cost_tracker.check_limits()
                if not limits["daily_ok"] or not limits["monthly_ok"]:
                    raise Exception("超出成本限制")
                
                start_time = time.time()
                
                # 发送请求
                response = await self.client.chat.completions.create(
                    model=self.config["llm"]["model"],
                    messages=messages,
                    **self.config["llm"]["parameters"]
                )
                
                processing_time = time.time() - start_time
                
                # 计算成本
                cost = self._calculate_cost(response.usage)
                
                # 更新统计和成本跟踪
                self.cost_tracker.add_cost(cost, response.usage.total_tokens)
                self.stats["total_requests"] += 1
                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += response.usage.total_tokens
                self.stats["total_cost"] += cost
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage,
                    "cost": cost,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                self.stats["total_requests"] += 1
                self.stats["failed_requests"] += 1
                logger.error(f"API请求失败: {e}")
                raise
    
    def _calculate_cost(self, usage) -> float:
        """
        计算API调用成本
        
        Args:
            usage: OpenAI使用情况对象
            
        Returns:
            成本（美元）
        """
        # GPT-4o-mini 价格 (2024年价格)
        input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
        output_cost_per_1k = 0.0006  # $0.60 per 1M tokens
        
        input_cost = (usage.prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.completion_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    async def clean_text(self, title: str, content: str) -> ProcessingResult:
        """
        清洗文本
        
        Args:
            title: 文章标题
            content: 文章内容
            
        Returns:
            处理结果
        """
        if not self.config["preprocessing_tasks"]["text_cleaning"]["enabled"]:
            return ProcessingResult(
                success=True,
                original_text=f"{title}\n{content}",
                processed_text=f"{title}\n{content}"
            )
        
        # 检查缓存
        cache_key = f"{title}\n{content}"
        cached_result = self.cache_manager.get(cache_key, "text_cleaning")
        if cached_result:
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                processed_text=cached_result["processed_text"]
            )
        
        try:
            system_prompt = self.config["preprocessing_tasks"]["text_cleaning"]["system_prompt"]
            user_prompt = self.config["preprocessing_tasks"]["text_cleaning"]["user_prompt_template"].format(
                title=title, content=content
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_request(messages, "text_cleaning")
            
            # 缓存结果
            cache_result = {"processed_text": response["content"]}
            self.cache_manager.set(cache_key, "text_cleaning", cache_result)
            
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                processed_text=response["content"],
                processing_time=response["processing_time"],
                tokens_used=response["usage"].total_tokens
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                original_text=cache_key,
                error_message=str(e)
            )
    
    async def assess_quality(self, title: str, content: str) -> ProcessingResult:
        """
        评估文本质量
        
        Args:
            title: 文章标题
            content: 文章内容
            
        Returns:
            处理结果
        """
        if not self.config["preprocessing_tasks"]["quality_assessment"]["enabled"]:
            return ProcessingResult(
                success=True,
                original_text=f"{title}\n{content}",
                quality_score=80.0  # 默认分数
            )
        
        cache_key = f"{title}\n{content}"
        cached_result = self.cache_manager.get(cache_key, "quality_assessment")
        if cached_result:
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                quality_score=cached_result["quality_score"]
            )
        
        try:
            system_prompt = self.config["preprocessing_tasks"]["quality_assessment"]["system_prompt"]
            user_prompt = self.config["preprocessing_tasks"]["quality_assessment"]["user_prompt_template"].format(
                title=title, content=content
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_request(messages, "quality_assessment")
            
            # 解析JSON响应
            quality_data = json.loads(response["content"])
            quality_score = quality_data.get("overall_score", 0)
            
            # 缓存结果
            cache_result = {"quality_score": quality_score}
            self.cache_manager.set(cache_key, "quality_assessment", cache_result)
            
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                quality_score=quality_score,
                processing_time=response["processing_time"],
                tokens_used=response["usage"].total_tokens
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                original_text=cache_key,
                error_message=str(e)
            )
    
    async def auto_label(self, title: str, content: str) -> ProcessingResult:
        """
        自动标注文本
        
        Args:
            title: 文章标题
            content: 文章内容
            
        Returns:
            处理结果
        """
        if not self.config["preprocessing_tasks"]["auto_labeling"]["enabled"]:
            return ProcessingResult(
                success=True,
                original_text=f"{title}\n{content}",
                category="未知",
                confidence=0.5
            )
        
        cache_key = f"{title}\n{content}"
        cached_result = self.cache_manager.get(cache_key, "auto_labeling")
        if cached_result:
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                category=cached_result["category"],
                confidence=cached_result["confidence"],
                reasoning=cached_result.get("reasoning")
            )
        
        try:
            system_prompt = self.config["preprocessing_tasks"]["auto_labeling"]["system_prompt"]
            user_prompt = self.config["preprocessing_tasks"]["auto_labeling"]["user_prompt_template"].format(
                title=title, content=content
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self._make_api_request(messages, "auto_labeling")
            
            # 解析JSON响应
            label_data = json.loads(response["content"])
            category = label_data.get("category", "未知")
            confidence = label_data.get("confidence", 0.0)
            reasoning = label_data.get("reasoning", "")
            
            # 缓存结果
            cache_result = {
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning
            }
            self.cache_manager.set(cache_key, "auto_labeling", cache_result)
            
            return ProcessingResult(
                success=True,
                original_text=cache_key,
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                processing_time=response["processing_time"],
                tokens_used=response["usage"].total_tokens
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                original_text=cache_key,
                error_message=str(e)
            )
    
    async def process_single_article(
        self, 
        title: str, 
        content: str, 
        tasks: List[str] = ["clean", "quality", "label"]
    ) -> Dict[str, ProcessingResult]:
        """
        处理单篇文章
        
        Args:
            title: 文章标题
            content: 文章内容
            tasks: 要执行的任务列表
            
        Returns:
            各任务的处理结果
        """
        results = {}
        
        if "clean" in tasks:
            results["clean"] = await self.clean_text(title, content)
        
        if "quality" in tasks:
            results["quality"] = await self.assess_quality(title, content)
        
        if "label" in tasks:
            results["label"] = await self.auto_label(title, content)
        
        return results
    
    async def process_batch(
        self,
        articles: List[Dict[str, str]],
        tasks: List[str] = ["clean", "quality", "label"],
        batch_size: int = None
    ) -> Tuple[List[Dict[str, ProcessingResult]], BatchProcessingStats]:
        """
        批量处理文章
        
        Args:
            articles: 文章列表，每个元素包含title和content
            tasks: 要执行的任务列表
            batch_size: 批次大小
            
        Returns:
            处理结果列表和统计信息
        """
        if batch_size is None:
            batch_size = self.config.get("batch_processing", {}).get("batch_size", 20)
        
        start_time = time.time()
        all_results = []
        total_tokens = 0
        total_cost = 0.0
        successful_count = 0
        quality_scores = []
        
        # 分批处理
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_tasks = []
            
            for article in batch:
                title = article.get("title", "")
                content = article.get("content", "")
                task = self.process_single_article(title, content, tasks)
                batch_tasks.append(task)
            
            # 并发执行批次任务
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    all_results.append({
                        "error": str(result)
                    })
                else:
                    all_results.append(result)
                    
                    # 统计成功的结果
                    if "quality" in result and result["quality"].success:
                        successful_count += 1
                        if result["quality"].quality_score is not None:
                            quality_scores.append(result["quality"].quality_score)
                    
                    # 累计token和成本
                    for task_result in result.values():
                        if hasattr(task_result, 'tokens_used') and task_result.tokens_used:
                            total_tokens += task_result.tokens_used
                        # 成本会在_make_api_request中自动跟踪
            
            # 进度报告
            logger.info(f"已处理 {min(i + batch_size, len(articles))}/{len(articles)} 篇文章")
        
        # 生成统计信息
        processing_time = time.time() - start_time
        total_cost = self.cost_tracker.get_daily_cost()  # 获取今日成本
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        stats = BatchProcessingStats(
            total_samples=len(articles),
            successful_samples=successful_count,
            failed_samples=len(articles) - successful_count,
            total_tokens_used=total_tokens,
            total_cost=total_cost,
            processing_time=processing_time,
            average_quality_score=average_quality
        )
        
        return all_results, stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            **self.stats,
            "daily_cost": self.cost_tracker.get_daily_cost(),
            "monthly_cost": self.cost_tracker.get_monthly_cost(),
            "cost_limits": self.cost_tracker.check_limits()
        }
    
    async def close(self):
        """关闭客户端"""
        await self.client.close()


async def main():
    """测试函数"""
    # 创建处理器
    processor = LLMProcessor()
    
    # 测试文章
    test_articles = [
        {
            "title": "政府发布新的经济政策",
            "content": "政府今日宣布了一系列新的经济刺激政策，旨在促进经济增长..."
        },
        {
            "title": "苹果公司发布新iPhone",
            "content": "苹果公司在今天的发布会上展示了最新的iPhone系列手机..."
        }
    ]
    
    try:
        # 批量处理
        results, stats = await processor.process_batch(test_articles)
        
        # 打印结果
        for i, result in enumerate(results):
            print(f"\n文章 {i+1} 处理结果:")
            for task, task_result in result.items():
                if task_result.success:
                    print(f"  {task}: 成功")
                    if task_result.processed_text:
                        print(f"    处理后文本: {task_result.processed_text[:100]}...")
                    if task_result.quality_score:
                        print(f"    质量分数: {task_result.quality_score}")
                    if task_result.category:
                        print(f"    分类: {task_result.category} (置信度: {task_result.confidence})")
                else:
                    print(f"  {task}: 失败 - {task_result.error_message}")
        
        # 打印统计信息
        print(f"\n处理统计:")
        print(f"  总样本数: {stats.total_samples}")
        print(f"  成功数: {stats.successful_samples}")
        print(f"  失败数: {stats.failed_samples}")
        print(f"  总耗时: {stats.processing_time:.2f}秒")
        print(f"  总成本: ${stats.total_cost:.4f}")
        print(f"  平均质量分数: {stats.average_quality_score:.2f}")
        
    finally:
        await processor.close()


if __name__ == "__main__":
    asyncio.run(main())
