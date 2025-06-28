# 负责任的新闻网络爬虫
# 
# 功能:
# - 遵守robots.txt协议
# - 速率限制和礼貌访问
# - 错误处理和重试机制
# - 数据质量控制
# - 支持多种数据源

# 加载必要的包
suppressPackageStartupMessages({
  library(rvest)
  library(httr)
  library(polite)
  library(robotstxt)
  library(jsonlite)
  library(dplyr)
  library(stringr)
  library(lubridate)
  library(purrr)
  library(xml2)
  library(curl)
  library(yaml)
})

# 设置全局选项
options(timeout = 30)  # 30秒超时
Sys.setlocale("LC_CTYPE", "UTF-8")  # 支持中文

# 日志记录函数
log_message <- function(message, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, message))
}

# 配置类
ScraperConfig <- list(
  # 默认配置
  default_delay = 5,  # 默认延迟秒数
  max_retries = 3,    # 最大重试次数
  timeout = 30,       # 请求超时
  user_agent = "Academic Research Bot - Text Classification Study",
  
  # 数据质量控制
  min_content_length = 50,
  max_content_length = 10000,
  
  # 输出配置
  output_dir = "data/raw/",
  log_file = "logs/scraper.log"
)

# 创建输出目录
create_directories <- function(config) {
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(config$log_file), recursive = TRUE, showWarnings = FALSE)
}

# 检查robots.txt合规性
check_robots_compliance <- function(url, user_agent = "*") {
  tryCatch({
    base_url <- str_extract(url, "https?://[^/]+")
    allowed <- robotstxt::paths_allowed(
      paths = url,
      domain = base_url,
      bot = user_agent
    )
    return(allowed)
  }, error = function(e) {
    log_message(paste("检查robots.txt失败:", e$message), "WARNING")
    return(FALSE)  # 保守策略，如果无法检查则不允许
  })
}

# 创建礼貌的会话
create_polite_session <- function(base_url, config) {
  tryCatch({
    session <- bow(
      url = base_url,
      user_agent = config$user_agent,
      delay = config$default_delay,
      verbose = TRUE
    )
    return(session)
  }, error = function(e) {
    log_message(paste("创建会话失败:", e$message), "ERROR")
    return(NULL)
  })
}

# 文本清洗函数
clean_text <- function(text) {
  if (is.null(text) || length(text) == 0 || is.na(text)) {
    return("")
  }
  
  # 移除HTML标签
  text <- str_replace_all(text, "<[^>]*>", "")
  
  # 移除多余的空白
  text <- str_replace_all(text, "\\s+", " ")
  
  # 移除首尾空白
  text <- str_trim(text)
  
  # 移除特殊字符（保留中文、英文、数字、基本标点）
  text <- str_replace_all(text, "[^\\p{L}\\p{N}\\p{P}\\p{Z}]", "")
  
  return(text)
}

# 验证文章质量
validate_article <- function(article, config) {
  # 检查必需字段
  required_fields <- c("title", "content", "url")
  for (field in required_fields) {
    if (is.null(article[[field]]) || article[[field]] == "") {
      return(list(valid = FALSE, reason = paste("缺少字段:", field)))
    }
  }
  
  # 检查内容长度
  content_length <- nchar(article$content)
  if (content_length < config$min_content_length) {
    return(list(valid = FALSE, reason = "内容过短"))
  }
  
  if (content_length > config$max_content_length) {
    return(list(valid = FALSE, reason = "内容过长"))
  }
  
  # 检查标题合理性
  title_length <- nchar(article$title)
  if (title_length < 5 || title_length > 200) {
    return(list(valid = FALSE, reason = "标题长度不合理"))
  }
  
  return(list(valid = TRUE, reason = ""))
}

# 通用文章提取器
extract_article_generic <- function(html_content, url) {
  tryCatch({
    # 尝试多种选择器策略
    title_selectors <- c(
      "h1",
      "[data-testid='headline']",
      ".headline",
      ".title",
      ".article-title",
      "title"
    )
    
    content_selectors <- c(
      ".article-body p",
      ".content p",
      ".post-content p",
      "[data-testid='article-body'] p",
      "article p",
      ".entry-content p"
    )
    
    # 提取标题
    title <- ""
    for (selector in title_selectors) {
      title_nodes <- html_nodes(html_content, selector)
      if (length(title_nodes) > 0) {
        title <- html_text(title_nodes[1])
        title <- clean_text(title)
        if (title != "") break
      }
    }
    
    # 提取正文
    content_paragraphs <- c()
    for (selector in content_selectors) {
      content_nodes <- html_nodes(html_content, selector)
      if (length(content_nodes) > 0) {
        paragraphs <- html_text(content_nodes)
        paragraphs <- map_chr(paragraphs, clean_text)
        paragraphs <- paragraphs[paragraphs != ""]
        if (length(paragraphs) > 0) {
          content_paragraphs <- paragraphs
          break
        }
      }
    }
    
    # 组合正文
    content <- paste(content_paragraphs, collapse = "\n")
    
    # 提取发布时间
    publish_date <- tryCatch({
      date_selectors <- c(
        "time[datetime]",
        ".publish-date",
        ".date",
        "[data-testid='timestamp']"
      )
      
      date_text <- ""
      for (selector in date_selectors) {
        date_nodes <- html_nodes(html_content, selector)
        if (length(date_nodes) > 0) {
          date_text <- html_attr(date_nodes[1], "datetime") %||% html_text(date_nodes[1])
          if (!is.null(date_text) && date_text != "") break
        }
      }
      
      if (date_text != "") {
        parsed_date <- ymd_hms(date_text, quiet = TRUE) %||% ymd(date_text, quiet = TRUE)
        if (!is.na(parsed_date)) {
          return(as.character(parsed_date))
        }
      }
      
      return(as.character(Sys.Date()))
    }, error = function(e) {
      return(as.character(Sys.Date()))
    })
    
    # 返回文章对象
    article <- list(
      title = title,
      content = content,
      url = url,
      publish_date = publish_date,
      word_count = length(strsplit(content, "\\s+")[[1]]),
      scrape_timestamp = as.character(Sys.time()),
      source = "generic_extractor"
    )
    
    return(article)
    
  }, error = function(e) {
    log_message(paste("文章提取失败:", e$message), "ERROR")
    return(NULL)
  })
}

# 特定网站的提取器
extract_article_news_api <- function(article_data) {
  tryCatch({
    article <- list(
      title = clean_text(article_data$title %||% ""),
      content = clean_text(article_data$content %||% article_data$description %||% ""),
      url = article_data$url %||% "",
      publish_date = article_data$publishedAt %||% as.character(Sys.Date()),
      source = "news_api",
      author = article_data$author %||% "",
      source_name = article_data$source$name %||% "",
      scrape_timestamp = as.character(Sys.time())
    )
    
    article$word_count <- length(strsplit(article$content, "\\s+")[[1]])
    
    return(article)
  }, error = function(e) {
    log_message(paste("News API文章解析失败:", e$message), "ERROR")
    return(NULL)
  })
}

# 从RSS源获取文章
scrape_rss_feed <- function(rss_url, config, max_articles = 50) {
  log_message(paste("开始处理RSS源:", rss_url))
  
  articles <- list()
  
  tryCatch({
    # 检查robots.txt
    if (!check_robots_compliance(rss_url)) {
      log_message(paste("RSS源不被robots.txt允许:", rss_url), "WARNING")
      return(articles)
    }
    
    # 获取RSS内容
    response <- GET(rss_url, user_agent(config$user_agent), timeout(config$timeout))
    
    if (status_code(response) != 200) {
      log_message(paste("RSS请求失败，状态码:", status_code(response)), "ERROR")
      return(articles)
    }
    
    # 解析RSS
    rss_content <- content(response, "text", encoding = "UTF-8")
    rss_xml <- read_xml(rss_content)
    
    # 提取文章链接
    items <- xml_find_all(rss_xml, ".//item")
    
    for (i in seq_len(min(length(items), max_articles))) {
      item <- items[[i]]
      
      title <- xml_text(xml_find_first(item, ".//title"))
      link <- xml_text(xml_find_first(item, ".//link"))
      description <- xml_text(xml_find_first(item, ".//description"))
      pub_date <- xml_text(xml_find_first(item, ".//pubDate"))
      
      if (!is.na(link) && link != "") {
        article <- list(
          title = clean_text(title),
          content = clean_text(description),
          url = link,
          publish_date = pub_date %||% as.character(Sys.Date()),
          source = "rss_feed",
          scrape_timestamp = as.character(Sys.time())
        )
        
        article$word_count <- length(strsplit(article$content, "\\s+")[[1]])
        
        # 验证文章质量
        validation <- validate_article(article, config)
        if (validation$valid) {
          articles <- append(articles, list(article))
          log_message(paste("成功提取RSS文章:", substr(title, 1, 50)))
        } else {
          log_message(paste("RSS文章质量不合格:", validation$reason), "WARNING")
        }
      }
      
      # 延迟
      Sys.sleep(config$default_delay)
    }
    
  }, error = function(e) {
    log_message(paste("RSS处理失败:", e$message), "ERROR")
  })
  
  log_message(paste("RSS源处理完成，获得", length(articles), "篇文章"))
  return(articles)
}

# 从网页URL列表爬取文章
scrape_urls <- function(urls, config) {
  log_message(paste("开始爬取", length(urls), "个URL"))
  
  articles <- list()
  failed_urls <- c()
  
  for (i in seq_along(urls)) {
    url <- urls[i]
    log_message(paste("处理URL", i, "/", length(urls), ":", url))
    
    tryCatch({
      # 检查robots.txt合规性
      if (!check_robots_compliance(url)) {
        log_message(paste("URL不被robots.txt允许:", url), "WARNING")
        failed_urls <- c(failed_urls, url)
        next
      }
      
      # 添加延迟
      Sys.sleep(config$default_delay)
      
      # 获取网页内容
      response <- GET(
        url,
        user_agent(config$user_agent),
        timeout(config$timeout)
      )
      
      if (status_code(response) != 200) {
        log_message(paste("请求失败，状态码:", status_code(response)), "WARNING")
        failed_urls <- c(failed_urls, url)
        next
      }
      
      # 解析HTML
      html_content <- read_html(content(response, "text", encoding = "UTF-8"))
      
      # 提取文章内容
      article <- extract_article_generic(html_content, url)
      
      if (!is.null(article)) {
        # 验证文章质量
        validation <- validate_article(article, config)
        if (validation$valid) {
          articles <- append(articles, list(article))
          log_message(paste("成功提取文章:", substr(article$title, 1, 50)))
        } else {
          log_message(paste("文章质量不合格:", validation$reason), "WARNING")
          failed_urls <- c(failed_urls, url)
        }
      } else {
        log_message("文章提取失败", "WARNING")
        failed_urls <- c(failed_urls, url)
      }
      
    }, error = function(e) {
      log_message(paste("处理URL失败:", e$message), "ERROR")
      failed_urls <- c(failed_urls, url)
    })
  }
  
  log_message(paste("爬取完成，成功:", length(articles), "失败:", length(failed_urls)))
  
  return(list(
    articles = articles,
    failed_urls = failed_urls,
    success_rate = length(articles) / length(urls)
  ))
}

# 保存数据
save_articles <- function(articles, config, filename_prefix = "scraped_news") {
  if (length(articles) == 0) {
    log_message("没有文章需要保存", "WARNING")
    return(NULL)
  }
  
  # 创建输出目录
  create_directories(config)
  
  # 生成文件名
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
  # 保存为JSON格式
  json_file <- file.path(config$output_dir, paste0(filename_prefix, "_", timestamp, ".json"))
  write_json(articles, json_file, pretty = TRUE, auto_unbox = TRUE)
  log_message(paste("已保存JSON文件:", json_file))
  
  # 保存为CSV格式（便于查看）
  csv_file <- file.path(config$output_dir, paste0(filename_prefix, "_", timestamp, ".csv"))
  articles_df <- map_dfr(articles, function(article) {
    data.frame(
      title = article$title %||% "",
      content = substr(article$content %||% "", 1, 1000),  # 截断长内容
      url = article$url %||% "",
      publish_date = article$publish_date %||% "",
      word_count = article$word_count %||% 0,
      source = article$source %||% "",
      scrape_timestamp = article$scrape_timestamp %||% "",
      stringsAsFactors = FALSE
    )
  })
  
  write.csv(articles_df, csv_file, row.names = FALSE, fileEncoding = "UTF-8")
  log_message(paste("已保存CSV文件:", csv_file))
  
  # 生成统计报告
  generate_scraping_report(articles, config, timestamp)
  
  return(list(
    json_file = json_file,
    csv_file = csv_file,
    article_count = length(articles)
  ))
}

# 生成爬取报告
generate_scraping_report <- function(articles, config, timestamp) {
  report_file <- file.path(config$output_dir, paste0("scraping_report_", timestamp, ".txt"))
  
  # 计算统计信息
  total_articles <- length(articles)
  avg_word_count <- mean(sapply(articles, function(x) x$word_count %||% 0))
  sources <- table(sapply(articles, function(x) x$source %||% "unknown"))
  
  # 生成报告内容
  report_content <- c(
    "新闻爬取报告",
    "=" %>% rep(20) %>% paste(collapse = ""),
    "",
    paste("爬取时间:", timestamp),
    paste("总文章数:", total_articles),
    paste("平均字数:", round(avg_word_count, 0)),
    "",
    "数据源分布:",
    paste(names(sources), ":", sources, collapse = "\n"),
    "",
    "配置信息:",
    paste("延迟设置:", config$default_delay, "秒"),
    paste("最小内容长度:", config$min_content_length),
    paste("最大内容长度:", config$max_content_length),
    "",
    "文章标题示例:",
    paste(sapply(articles[1:min(5, length(articles))], function(x) 
      paste("-", substr(x$title, 1, 60))
    ), collapse = "\n")
  )
  
  writeLines(report_content, report_file, useBytes = TRUE)
  log_message(paste("已生成爬取报告:", report_file))
}

# 主函数
main_scraper <- function(config_file = NULL, source_type = "rss", source_list = NULL, max_articles = 100) {
  # 加载配置
  if (!is.null(config_file) && file.exists(config_file)) {
    config <- yaml::read_yaml(config_file)
    config <- modifyList(ScraperConfig, config)
  } else {
    config <- ScraperConfig
  }
  
  log_message("开始新闻爬取任务")
  log_message(paste("数据源类型:", source_type))
  
  all_articles <- list()
  
  if (source_type == "rss" && !is.null(source_list)) {
    # RSS源爬取
    for (rss_url in source_list) {
      articles <- scrape_rss_feed(rss_url, config, max_articles)
      all_articles <- append(all_articles, articles)
    }
  } else if (source_type == "urls" && !is.null(source_list)) {
    # URL列表爬取
    result <- scrape_urls(source_list, config)
    all_articles <- result$articles
  } else {
    log_message("无效的数据源配置", "ERROR")
    return(NULL)
  }
  
  # 保存结果
  if (length(all_articles) > 0) {
    save_result <- save_articles(all_articles, config)
    log_message(paste("爬取任务完成，共获得", length(all_articles), "篇文章"))
    return(save_result)
  } else {
    log_message("没有成功爬取到任何文章", "WARNING")
    return(NULL)
  }
}

# 示例用法
if (!interactive()) {
  # 命令行参数解析
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) > 0) {
    source_type <- args[1]
    
    if (source_type == "rss") {
      # 示例RSS源（替换为实际可用的RSS源）
      rss_sources <- c(
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.bbci.co.uk/news/rss.xml"
      )
      
      result <- main_scraper(
        source_type = "rss",
        source_list = rss_sources,
        max_articles = 20
      )
      
    } else if (source_type == "demo") {
      # 演示模式 - 创建模拟数据
      demo_articles <- list(
        list(
          title = "政府发布新的经济政策促进发展",
          content = "政府今日发布了一系列新的经济政策，旨在促进国内经济发展，提高就业率，改善民生。这些政策包括减税措施、基础设施投资计划等多项内容。",
          url = "http://example.com/news1",
          publish_date = as.character(Sys.Date()),
          source = "demo",
          word_count = 45,
          scrape_timestamp = as.character(Sys.time())
        ),
        list(
          title = "科技公司推出革命性人工智能产品",
          content = "某知名科技公司今天宣布推出其最新的人工智能产品，该产品采用了先进的深度学习技术，能够显著提高工作效率。业界专家认为这将改变整个行业的格局。",
          url = "http://example.com/news2",
          publish_date = as.character(Sys.Date()),
          source = "demo",
          word_count = 52,
          scrape_timestamp = as.character(Sys.time())
        ),
        list(
          title = "股市创新高投资者信心增强",
          content = "今日股市表现强劲，主要指数创下历史新高。分析师表示，这主要得益于近期发布的积极经济数据和企业盈利预期的改善。投资者信心明显增强，市场交易活跃。",
          url = "http://example.com/news3",
          publish_date = as.character(Sys.Date()),
          source = "demo",
          word_count = 48,
          scrape_timestamp = as.character(Sys.time())
        )
      )
      
      config <- ScraperConfig
      result <- save_articles(demo_articles, config, "demo_news")
      
      if (!is.null(result)) {
        log_message(paste("演示数据已生成:", result$json_file))
      }
    } else {
      cat("用法: Rscript web_scraper.R [rss|demo]\n")
      cat("  rss  - 从RSS源爬取新闻\n")
      cat("  demo - 生成演示数据\n")
    }
  } else {
    cat("新闻爬虫使用说明:\n")
    cat("Rscript web_scraper.R demo  # 生成演示数据\n")
    cat("Rscript web_scraper.R rss   # 从RSS源爬取\n")
  }
}
