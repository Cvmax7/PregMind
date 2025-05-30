# 爱丁堡产后抑郁量表(EPDS)评估智能体

基于微调后的LLaMA3-8B模型的产后抑郁风险自动化评估与分级系统，通过结构化Prompt设计和多轮对话判断机制，引导孕产妇完成EPDS评估，生成专业的风险评估报告。

## 项目简介

爱丁堡产后抑郁量表(Edinburgh Postnatal Depression Scale, EPDS)是国际公认的产后抑郁筛查工具，包含10个问题，每个问题有4个选项，总分30分。本智能体基于经过心理健康领域数据微调的LLaMA3-8B模型，提供智能化的EPDS评估流程，具有以下特点：

- **智能解释用户回答**：能够理解用户的自然语言回答，并准确匹配到EPDS量表的选项
- **风险分级评估**：根据得分自动划分为低风险、轻度风险、中度风险、高风险四个等级
- **自杀风险检测**：特别关注问题10(自伤想法)的回答，及时发现自杀风险
- **专业评估报告**：生成包含风险级别、推荐措施的专业评估报告
- **双重交互方式**：同时支持命令行和Web界面两种交互模式
- **多种报告格式**：支持JSON和PDF两种报告格式导出


## 系统功能详解

### 1. EPDS量表评估流程

系统通过10个问题依次引导用户，每个问题有4个选项(0-3分)。用户可以直接回答选项编号(1-4)或用自然语言描述自己的感受，系统会智能解释并匹配到最合适的选项。

### 2. 风险分级标准

根据EPDS总分(满分30分)，系统将风险分为4个等级：

- **低风险** (0-9分): 当前可能没有明显的产后抑郁症状
- **轻度风险** (10-12分): 可能有轻度的产后抑郁症状，建议关注情绪变化
- **中度风险** (13-16分): 可能有中度的产后抑郁症状，建议寻求专业支持
- **高风险** (17-30分): 可能有严重的产后抑郁症状，强烈建议尽快咨询专业医生

### 3. 自杀风险检测

系统特别关注问题10 "在过去的7天里，您是否曾想过要伤害自己？"的回答。如果得分≥1分，系统会在评估报告中提供特别警示和建议。

### 4. 报告导出功能

支持两种格式的评估报告导出：

- **JSON格式**: 包含完整的评估数据，方便数据分析和系统集成
- **PDF格式**: 生成专业的评估报告文档，适合打印和分享


## 微调模型说明

本系统使用经过心理健康领域数据微调的LLaMA3-8B模型。微调过程使用了PsyQA心理健康问答数据集，通过LoRA技术在MindSpore环境下完成，具体流程见`Agent_EPDS/EPDS.ipynb`。

## 注意事项

- 本系统仅用于辅助筛查，不能替代专业医疗诊断
- 评估结果应由专业医疗人员进一步确认
- 如检测到自杀风险，应立即寻求专业帮助
- 首次运行时，模型加载可能需要较长时间
