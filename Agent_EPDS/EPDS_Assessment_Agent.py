import os
import json
import numpy as np
import time
import mindspore as ms
from mindspore import Tensor
from mindformers import LlamaTokenizer, LlamaForCausalLM, TextGenerationConfig

class DialogueNode:
    """对话节点基类，表示EPDS评估流程中的一个交互环节"""
    def __init__(self, assessment_agent):
        self.assessment_agent = assessment_agent
    
    def process(self, input_data=None):
        """处理当前节点逻辑，返回结果和下一个节点"""
        pass


class GuidedInterviewNode(DialogueNode):
    """
    1.1 引导访谈节点
    负责通过结构化Prompt与用户进行对话式提问，引导用户完成EPDS评估
    """
    def __init__(self, assessment_agent):
        super().__init__(assessment_agent)
        self.current_question_idx = 0
        
    def _create_interview_prompt(self, question_data):
        """创建专业访谈风格的提问提示词"""
        question = question_data["question"]
        options = question_data["options"]
        
        # 创建更具专业性和心理安全的提问格式
        prompt_parts = [
            f"问题 {question_data['id']}/10: {question}",
            "在过去的7天里，请描述您的真实感受:",
        ]
        
        # 添加选项
        for i, option in enumerate(options):
            prompt_parts.append(f"{i+1}. {option['text']}")
        
        # 添加引导语
        prompt_parts.append("\n请以自然方式描述您的感受，或直接选择最符合的选项(1-4)。")
        
        return "\n".join(prompt_parts)
    
    def process(self, input_data=None):
        """处理当前问题，返回提问内容"""
        if self.current_question_idx >= len(self.assessment_agent.epds_questions):
            # 所有问题已完成，进入报告生成节点
            return None, "ReportGenerationNode"
        
        # 获取当前问题
        current_q = self.assessment_agent.epds_questions[self.current_question_idx]
        
        # 创建专业访谈风格的提问
        interview_prompt = self._create_interview_prompt(current_q)
        
        # 记录到对话历史
        self.assessment_agent.conversation_history.append({"role": "assistant", "content": interview_prompt})
        
        # 返回提问内容和下一个节点名称
        return interview_prompt, "ResponseEvaluationNode"


class ResponseEvaluationNode(DialogueNode):
    """
    1.2 判断回复节点
    负责解析和评估用户的回答，确定对应的EPDS选项和分数
    """
    def __init__(self, assessment_agent):
        super().__init__(assessment_agent)
    
    def _generate_clarification_prompt(self, user_answer, question_data):
        """生成澄清提示，当用户回答不明确时使用"""
        return f"""我理解您的回答是: "{user_answer}"

为了准确评估，我需要更明确地了解您的感受。
问题是: "{question_data['question']}"

您的回答最接近以下哪一项?
{self.assessment_agent._format_options(question_data['options'])}

请选择一个最符合您情况的选项，或提供更多细节。"""
    
    def _needs_clarification(self, answer_text, matched_option, confidence):
        """判断是否需要澄清用户回答"""
        # 如果回答过短或过于模糊
        if len(answer_text.strip()) < 3 and not answer_text.strip() in ["1", "2", "3", "4"]:
            return True
        
        # 如果置信度过低
        if confidence < 0.7:  # 假设置信度阈值为0.7
            return True
            
        return False
    
    def _match_option_with_confidence(self, answer_text, question_data):
        """匹配选项并返回置信度"""
        options = question_data["options"]
        
        # 尝试匹配数字选择
        if answer_text.strip() in ["1", "2", "3", "4"]:
            option_idx = int(answer_text.strip()) - 1
            if 0 <= option_idx < len(options):
                return options[option_idx], 1.0  # 数字选择的置信度最高
        
        # 尝试直接匹配选项文本
        for option in options:
            if option["text"].lower() in answer_text.lower():
                # 计算匹配程度
                match_ratio = len(option["text"]) / len(answer_text) if len(answer_text) > 0 else 0
                confidence = min(match_ratio * 1.5, 1.0)  # 调整置信度
                return option, confidence
        
        # 使用模型进行语义分析
        interpretation_prompt = f"""基于用户的回答: "{answer_text}"
请分析这个回答最接近问题"{question_data['question']}"的哪个选项:
{self.assessment_agent._format_options(options)}

请输出JSON格式:
{{"option_number": 选项编号(1-4), "confidence": 匹配置信度(0.0-1.0)}}"""
        
        # 使用模型解释
        interpretation = self.assessment_agent.generate_response(interpretation_prompt)
        
        # 尝试解析JSON结果
        try:
            import re
            json_match = re.search(r'\{.*\}', interpretation)
            if json_match:
                result = json.loads(json_match.group(0))
                option_idx = int(result.get("option_number", 1)) - 1
                confidence = float(result.get("confidence", 0.5))
                
                if 0 <= option_idx < len(options):
                    return options[option_idx], confidence
        except:
            # 解析失败，回退到简单匹配
            for i in range(1, 5):
                if str(i) in interpretation[:10]:
                    option_idx = i - 1
                    if 0 <= option_idx < len(options):
                        return options[option_idx], 0.5
        
        # 默认返回第一个选项和低置信度
        return options[0], 0.3
    
    def process(self, input_data):
        """
        处理用户回答，确定选择的选项和分数
        
        参数:
            input_data: 用户回答文本
        
        返回值:
            (result, next_node): 处理结果和下一个节点名称
        """
        # 记录用户回答
        self.assessment_agent.conversation_history.append({"role": "user", "content": input_data})
        
        # 获取当前问题
        current_question_idx = self.assessment_agent.guided_interview_node.current_question_idx
        current_q = self.assessment_agent.epds_questions[current_question_idx]
        
        # 匹配选项并返回置信度
        matched_option, confidence = self._match_option_with_confidence(input_data, current_q)
        
        # 判断是否需要澄清
        if self._needs_clarification(input_data, matched_option, confidence):
            # 需要澄清
            clarification_prompt = self._generate_clarification_prompt(input_data, current_q)
            self.assessment_agent.conversation_history.append({"role": "assistant", "content": clarification_prompt})
            return clarification_prompt, "ResponseEvaluationNode"  # 继续在此节点处理澄清回答
        
        # 不需要澄清，确定选项和分数
        selected_option = matched_option["text"]
        score = matched_option["score"]
        
        # 返回确认信息和下一个节点
        confirmation = f"我了解了，您的感受是: {selected_option}"
        return (current_question_idx, selected_option, score, confirmation), "ResultUpdateNode"


class ResultUpdateNode(DialogueNode):
    """
    1.3 结果更新节点
    负责更新评估结果和分数
    """
    def __init__(self, assessment_agent):
        super().__init__(assessment_agent)
    
    def process(self, input_data):
        """
        更新评估结果
        
        参数:
            input_data: (question_idx, selected_option, score, confirmation)
        
        返回值:
            (confirmation, next_node): 确认消息和下一个节点名称
        """
        question_idx, selected_option, score, confirmation = input_data
        
        # 记录回答和得分
        while len(self.assessment_agent.answers) <= question_idx:
            self.assessment_agent.answers.append(None)
            self.assessment_agent.scores.append(None)
            
        self.assessment_agent.answers[question_idx] = selected_option
        self.assessment_agent.scores[question_idx] = score
        
        # 更新对话历史
        self.assessment_agent.conversation_history.append({"role": "assistant", "content": confirmation})
        
        # 移动到下一个问题
        self.assessment_agent.guided_interview_node.current_question_idx += 1
        
        # 判断是否完成所有问题
        if self.assessment_agent.guided_interview_node.current_question_idx >= len(self.assessment_agent.epds_questions):
            return confirmation, "ReportGenerationNode"
        else:
            return confirmation, "GuidedInterviewNode"


class ReportGenerationNode(DialogueNode):
    """
    1.4 生成格式化报告节点
    生成专业的评估报告，包含风险评级和建议
    """
    def __init__(self, assessment_agent):
        super().__init__(assessment_agent)
    
    def _analyze_emotional_patterns(self):
        """分析情绪波动模式"""
        patterns = []
        
        # 问题1-2分析积极情绪能力
        positive_scores = self.assessment_agent.scores[0:2]
        if sum(positive_scores) >= 4:
            patterns.append("积极情绪体验能力显著降低，可能难以从日常生活中获得愉悦感。")
        elif sum(positive_scores) >= 2:
            patterns.append("积极情绪体验有所减弱，但仍保留部分愉悦体验能力。")
            
        # 问题3-6分析焦虑和压力感知
        anxiety_scores = self.assessment_agent.scores[2:6]
        if sum(anxiety_scores) >= 8:
            patterns.append("焦虑水平显著升高，伴随压力应对能力下降，需要重点关注。")
        elif sum(anxiety_scores) >= 4:
            patterns.append("存在一定程度的焦虑和压力感知，但仍在可控范围内。")
            
        # 问题7-9分析抑郁核心症状
        depression_scores = self.assessment_agent.scores[6:9]
        if sum(depression_scores) >= 6:
            patterns.append("核心抑郁症状明显，包括情绪低落、睡眠困扰和哭泣倾向。")
        elif sum(depression_scores) >= 3:
            patterns.append("存在轻度抑郁症状，情绪波动增加但尚未达到临床显著水平。")
            
        # 如果没有显著问题
        if not patterns:
            patterns.append("各项情绪指标大多处于健康范围内，未发现显著的情绪困扰模式。")
            
        return patterns
    
    def _generate_intervention_suggestions(self, risk_level):
        """根据风险级别生成干预建议"""
        suggestions = [
            "与您的产科医生或全科医生分享这个评估结果，获取专业医疗建议。",
            "保持规律的睡眠、均衡的饮食和适度的身体活动，这对情绪健康至关重要。",
            "寻求社会支持，与伴侣、家人或朋友分享您的感受和困难。"
        ]
        
        if risk_level == "低风险":
            suggestions.extend([
                "建立日常情绪记录习惯，关注情绪变化并及时调整。",
                "学习简单的放松技巧，如深呼吸或正念冥想，帮助缓解日常压力。"
            ])
        elif risk_level == "轻度风险":
            suggestions.extend([
                "考虑参加产后支持小组，与其他新妈妈分享经验和情感支持。",
                "学习压力管理和情绪调节技巧，如认知行为疗法的基本原则。",
                "安排与心理健康专业人士的初步咨询，评估是否需要进一步支持。"
            ])
        elif risk_level == "中度风险":
            suggestions.extend([
                "建议尽快安排与精神健康专业人士（心理医生或精神科医生）的评估。",
                "探索心理治疗选项，如认知行为疗法(CBT)或人际关系心理治疗(IPT)。",
                "与伴侣或家人一起制定支持计划，分担育儿责任，确保您有足够休息时间。",
                "关注睡眠质量，睡眠问题可能加剧抑郁症状。"
            ])
        else:  # 高风险
            suggestions.extend([
                "请立即联系心理健康专业人士进行评估和治疗规划。",
                "与您的家庭医生讨论是否需要药物治疗和/或心理治疗的综合干预。",
                "确保家人了解您的状况，并积极参与支持网络的构建。",
                "制定安全计划，包括紧急联系人和应对情绪危机的步骤。",
                "优先考虑自我照顾，暂时减轻一些责任，专注于恢复。"
            ])
            
        return suggestions
    
    def _extract_key_expressions(self):
        """从对话历史中提取用户关键表述"""
        key_expressions = []
        
        # 情绪相关关键词
        emotion_keywords = [
            "悲伤", "难过", "焦虑", "担忧", "害怕", "恐惧", "压力", "疲惫", 
            "绝望", "无助", "无望", "痛苦", "烦躁", "愤怒", "内疚", "自责",
            "哭泣", "失眠", "睡不着", "不开心", "情绪低落", "抑郁"
        ]
        
        # 从对话历史中提取用户发言
        for msg in self.assessment_agent.conversation_history:
            if msg["role"] == "user":
                user_text = msg["content"]
                
                # 检查是否包含情绪关键词
                for keyword in emotion_keywords:
                    if keyword in user_text and len(user_text) > 10:  # 确保不是简单的回答
                        # 将表述加入列表，避免重复
                        if user_text not in key_expressions:
                            key_expressions.append(user_text)
                            break
        
        # 如果没有找到关键表述，返回空列表
        return key_expressions[:3]  # 最多返回3条关键表述
    
    def _build_transfer_context(self, total_score, risk_level, emotional_patterns, key_expressions):
        """构建传递给Agent2的上下文"""
        transfer_context = {
            "assessment_result": {
                "total_score": total_score,
                "risk_level": risk_level,
                "emotional_patterns": emotional_patterns,
                "suicide_risk": self.assessment_agent._check_suicide_risk()[0]
            },
            "user_expressions": key_expressions,
            "assessment_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return transfer_context
    
    def _should_transfer_to_agent2(self, risk_level):
        """判断是否需要转接到Agent2"""
        # 如果风险等级为"轻度风险"及以上，建议转接
        return risk_level in ["轻度风险", "中度风险", "高风险"]
    
    def _generate_transfer_recommendation(self, risk_level, total_score):
        """生成转接到Agent2的建议文本"""
        if risk_level == "轻度风险":
            return f"""根据您的EPDS评估得分({total_score}分)，您可能正经历轻度的产后情绪困扰。

我们建议您与我们的心理健康支持顾问进行进一步交流，他/她可以提供更加个性化的情绪支持和应对策略。

您是否愿意转接到心理健康支持顾问？
1. 是，我愿意继续交流
2. 不，我现在只需要这份评估报告"""
            
        elif risk_level == "中度风险":
            return f"""根据您的EPDS评估得分({total_score}分)，您可能正经历中度的产后抑郁症状。

强烈建议您与我们的心理健康支持顾问进行深入交流，获取专业的情绪支持和干预策略，这对缓解您的情绪困扰非常重要。

您是否愿意转接到心理健康支持顾问？
1. 是，我愿意继续交流
2. 不，我现在只需要这份评估报告"""
            
        else:  # 高风险
            return f"""您的EPDS评估得分({total_score}分)表明您可能正经历严重的产后抑郁症状。

我们强烈建议您立即与我们的心理健康支持顾问交流，获取紧急支持和专业干预建议。同时，也请尽快联系您的医疗服务提供者。

您是否愿意转接到心理健康支持顾问？
1. 是，我愿意继续交流
2. 不，我现在只需要这份评估报告"""
    
    def process(self, input_data=None):
        """
        生成评估报告，并在需要时建议转接到Agent2
        
        参数:
            input_data: 如果是字符串，表示用户对转接建议的回应
                       如果为None，表示初次生成报告
        
        返回值:
            (response, next_node): 响应内容和下一个节点名称
        """
        # 如果评估尚未完成
        if len(self.assessment_agent.scores) < 10 or None in self.assessment_agent.scores:
            return "评估尚未完成，请回答所有问题以生成报告。", "GuidedInterviewNode"
        
        # 如果已经完成评估但用户尚未对转接建议做出回应
        if self.assessment_agent.assessment_complete and isinstance(input_data, str):
            # 处理用户对转接建议的回应
            if "1" in input_data or "是" in input_data or "愿意" in input_data:
                # 用户同意转接到Agent2
                total_score = sum(self.assessment_agent.scores)
                risk_level, _ = self.assessment_agent._calculate_risk_level(total_score)
                emotional_patterns = self._analyze_emotional_patterns()
                key_expressions = self._extract_key_expressions()
                
                # 构建转接上下文
                transfer_context = self._build_transfer_context(
                    total_score, risk_level, emotional_patterns, key_expressions
                )
                
                # 将上下文保存到assessment_agent中，以便外部程序访问
                self.assessment_agent.transfer_context = transfer_context
                self.assessment_agent.should_transfer = True
                
                return "正在为您转接到心理健康支持顾问，请稍候...", None
            else:
                # 用户拒绝转接，结束对话
                self.assessment_agent.should_transfer = False
                return "感谢您完成EPDS评估。如有需要，请随时联系专业医疗人员获取进一步帮助。", None
        
        # 生成初始评估报告
        # 计算总分
        total_score = sum(self.assessment_agent.scores)
        
        # 确定风险级别
        risk_level, risk_description = self.assessment_agent._calculate_risk_level(total_score)
        
        # 检查自杀风险
        suicide_risk, suicide_note = self.assessment_agent._check_suicide_risk()
        
        # 情绪波动解读
        emotional_patterns = self._analyze_emotional_patterns()
        
        # 干预建议
        intervention_suggestions = self._generate_intervention_suggestions(risk_level)
        
        # 构建专业格式的报告
        report_parts = [
            "## 爱丁堡产后抑郁量表(EPDS)评估报告",
            "",
            f"总分: {total_score}/30",
            "",
            f"风险级别: {risk_level}",
            "",
            "## 评估描述:",
            risk_description
        ]
        
        # 添加自杀风险警告（如果存在）
        if suicide_risk:
            report_parts.extend(["", "## ⚠️ 重要提示:", suicide_note])
        
        # 添加情绪波动分析
        report_parts.extend(["", "## 情绪状态分析:"])
        for pattern in emotional_patterns:
            report_parts.append(f"- {pattern}")
        
        # 添加干预建议
        report_parts.extend(["", "## 建议干预措施:"])
        for i, suggestion in enumerate(intervention_suggestions):
            report_parts.append(f"{i+1}. {suggestion}")
        
        # 添加免责声明
        report_parts.extend([
            "",
            "## 重要提示:",
            "本评估报告仅用于筛查目的，不构成医疗诊断。如有任何疑虑，请咨询专业医疗人员。"
        ])
        
        # 合并报告
        report = "\n".join(report_parts)
        
        # 更新对话历史和评估状态
        self.assessment_agent.conversation_history.append({"role": "assistant", "content": report})
        self.assessment_agent.assessment_complete = True
        
        # 判断是否需要转接到Agent2
        if self._should_transfer_to_agent2(risk_level):
            # 生成转接建议
            transfer_recommendation = self._generate_transfer_recommendation(risk_level, total_score)
            
            # 添加到对话历史
            self.assessment_agent.conversation_history.append({"role": "assistant", "content": transfer_recommendation})
            
            # 初始化转接相关属性
            self.assessment_agent.should_transfer = None  # 表示等待用户回应
            
            # 返回报告和转接建议
            return report + "\n\n" + transfer_recommendation, "ReportGenerationNode"
        else:
            # 风险较低，不需要转接
            self.assessment_agent.should_transfer = False
            return report, None


class EPDSAssessmentAgent:
    """
    爱丁堡产后抑郁量表(EPDS)评估智能体
    基于微调后的LLaMA3-8B模型，整合EPDS量表进行产后抑郁风险的自动化评估与分级
    """
    
    def __init__(self, model_path, device_target="Ascend", device_id=0):
        """
        初始化EPDS评估智能体
        
        参数:
            model_path: 微调后的模型路径
            device_target: 计算设备类型，如"Ascend"、"GPU"等
            device_id: 设备ID
        """
        # 配置MindSpore执行环境
        try:
            ms.set_context(mode=ms.GRAPH_MODE, device_target=device_target, device_id=device_id)
            
            # 初始化模型和分词器
            print(f"正在加载LLaMA3模型，路径：{model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self.model = LlamaForCausalLM.from_pretrained(model_path)
            
            # 设置文本生成配置
            self.gen_config = TextGenerationConfig(
                max_length=4096,
                top_k=5,
                top_p=0.85,
                temperature=0.7,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            print("模型加载成功!")
            
            # 验证模型是否正常工作
            self.validate_model()
        except FileNotFoundError as e:
            print(f"错误: {e}")
            raise
        except Exception as e:
            print(f"加载模型时出现错误: {e}")
            raise
        
        # 加载EPDS量表
        self.epds_questions = self._load_epds_questions()
        
        # 评估状态跟踪
        self.answers = []
        self.scores = []
        self.conversation_history = []
        self.assessment_complete = False
        
        # 初始化交互节点
        self.guided_interview_node = GuidedInterviewNode(self)
        self.response_evaluation_node = ResponseEvaluationNode(self)
        self.result_update_node = ResultUpdateNode(self)
        self.report_generation_node = ReportGenerationNode(self)
        
        # 流程节点映射
        self.nodes = {
            "GuidedInterviewNode": self.guided_interview_node,
            "ResponseEvaluationNode": self.response_evaluation_node,
            "ResultUpdateNode": self.result_update_node,
            "ReportGenerationNode": self.report_generation_node
        }
        
        # 当前节点
        self.current_node_name = "GuidedInterviewNode"
        
        # 多智能体协同相关属性
        self.should_transfer = False  # 是否需要转接到Agent2
        self.transfer_context = None  # 要传递给Agent2的上下文信息
    
    def validate_model(self):
        """验证模型是否正常工作"""
        try:
            test_messages = [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": "你好，请简单介绍一下爱丁堡产后抑郁量表。"}
            ]
            
            # 将消息转换为模型输入格式
            input_text = self.tokenizer.build_chat_input(test_messages)
            input_ids = input_text["input_ids"]
            
            # 生成回应
            output_ids = self.model.generate(
                Tensor(input_ids, ms.int32),
                max_length=100,  # 简短回应即可
                top_k=self.gen_config.top_k,
                top_p=self.gen_config.top_p,
                temperature=self.gen_config.temperature,
                repetition_penalty=self.gen_config.repetition_penalty,
                do_sample=self.gen_config.do_sample,
                eos_token_id=self.gen_config.eos_token_id
            )
            
            # 检查输出是否有效
            response = self.tokenizer.decode(output_ids[0].asnumpy().tolist(), skip_special_tokens=True)
            if len(response) < 10:  # 有效回应应该有一定长度
                raise Exception("模型生成的回应过短，可能不正常工作")
                
            print("模型验证成功，可以正常生成回应。")
        except Exception as e:
            print(f"模型验证失败: {e}")
            raise Exception(f"模型无法正常工作，请检查模型路径和配置: {e}")
    
    def _load_epds_questions(self):
        """加载EPDS量表问题及评分选项"""
        return [
            {
                "id": 1,
                "question": "在过去的7天里，您能够笑并且看到事物有趣的一面吗？",
                "options": [
                    {"text": "与以往一样", "score": 0},
                    {"text": "没有以往那么多", "score": 1},
                    {"text": "肯定比以往少", "score": 2},
                    {"text": "完全不能", "score": 3}
                ]
            },
            {
                "id": 2,
                "question": "在过去的7天里，您是否期待未来的愉快事情？",
                "options": [
                    {"text": "与以往一样", "score": 0},
                    {"text": "比以往稍微少一点", "score": 1},
                    {"text": "比以往明显少", "score": 2},
                    {"text": "几乎没有", "score": 3}
                ]
            },
            {
                "id": 3,
                "question": "在过去的7天里，当事情出错时，您是否会不必要地责备自己？",
                "options": [
                    {"text": "不，从不会", "score": 0},
                    {"text": "很少会", "score": 1},
                    {"text": "是的，有时会", "score": 2},
                    {"text": "是的，大多数时候会", "score": 3}
                ]
            },
            {
                "id": 4,
                "question": "在过去的7天里，您是否曾经无缘无故地感到焦虑或担忧？",
                "options": [
                    {"text": "不，完全没有", "score": 0},
                    {"text": "几乎没有", "score": 1},
                    {"text": "是的，有时候", "score": 2},
                    {"text": "是的，经常如此", "score": 3}
                ]
            },
            {
                "id": 5,
                "question": "在过去的7天里，您是否曾经无缘无故地感到害怕或惊慌？",
                "options": [
                    {"text": "不，完全没有", "score": 0},
                    {"text": "不，没有", "score": 1},
                    {"text": "是的，有时候", "score": 2},
                    {"text": "是的，相当经常", "score": 3}
                ]
            },
            {
                "id": 6,
                "question": "在过去的7天里，您是否感到事情太多而无法应对？",
                "options": [
                    {"text": "不，我一直能应付自如", "score": 0},
                    {"text": "不，大多数时候我都能应付得很好", "score": 1},
                    {"text": "是的，有时候我应付得不像往常那样好", "score": 2},
                    {"text": "是的，大多数时候我都无法应付", "score": 3}
                ]
            },
            {
                "id": 7,
                "question": "在过去的7天里，您是否因为不开心而难以入睡？",
                "options": [
                    {"text": "不，一点也没有", "score": 0},
                    {"text": "不是很频繁", "score": 1},
                    {"text": "是的，有时候", "score": 2},
                    {"text": "是的，大多数时候", "score": 3}
                ]
            },
            {
                "id": 8,
                "question": "在过去的7天里，您是否感到悲伤或难过？",
                "options": [
                    {"text": "不，一点也没有", "score": 0},
                    {"text": "不是很频繁", "score": 1},
                    {"text": "是的，相当频繁", "score": 2},
                    {"text": "是的，大多数时候", "score": 3}
                ]
            },
            {
                "id": 9,
                "question": "在过去的7天里，您是否因为不开心而哭泣？",
                "options": [
                    {"text": "不，从不", "score": 0},
                    {"text": "很少，只有一两次", "score": 1},
                    {"text": "是的，有时候", "score": 2},
                    {"text": "是的，大多数时候", "score": 3}
                ]
            },
            {
                "id": 10,
                "question": "在过去的7天里，您是否曾想过要伤害自己？",
                "options": [
                    {"text": "从未想过", "score": 0},
                    {"text": "很少", "score": 1},
                    {"text": "有时候", "score": 2},
                    {"text": "是的，相当频繁", "score": 3}
                ]
            }
        ]
    
    def _format_options(self, options):
        """格式化问题选项，用于提示词"""
        return "\n".join([f"{i+1}. {option['text']}" for i, option in enumerate(options)])
    
    def _create_system_prompt(self):
        """创建系统提示词"""
        return """你是一位专业的产后抑郁筛查助手，负责协助完成爱丁堡产后抑郁量表(EPDS)评估。
你的任务是以温和、专业的方式引导用户完成10个问题的评估，并根据他们的回答计算得分。
请保持对话友好且支持性，理解用户可能正在经历的情感困扰。
在解释每个问题时，确保用户理解问题的含义，但不要引导他们选择特定答案。
如果用户的回答不明确，请温和地询问更多细节以确定最适合的选项。
请记住，你的目标是准确评估，而不是诊断或治疗。"""
    
    def _format_conversation(self):
        """格式化对话历史，用于提示模型"""
        messages = [{"role": "system", "content": self._create_system_prompt()}]
        
        for msg in self.conversation_history:
            messages.append(msg)
        
        return messages
    
    def generate_response(self, prompt):
        """使用模型生成回应"""
        messages = self._format_conversation()
        messages.append({"role": "user", "content": prompt})
        
        # 将消息转换为模型输入格式
        input_text = self.tokenizer.build_chat_input(messages)
        input_ids = input_text["input_ids"]
        
        # 生成回应
        output_ids = self.model.generate(
            Tensor(input_ids, ms.int32),
            max_length=self.gen_config.max_length,
            top_k=self.gen_config.top_k,
            top_p=self.gen_config.top_p,
            temperature=self.gen_config.temperature,
            repetition_penalty=self.gen_config.repetition_penalty,
            do_sample=self.gen_config.do_sample,
            eos_token_id=self.gen_config.eos_token_id
        )
        
        # 解码输出
        response = self.tokenizer.decode(output_ids[0].asnumpy().tolist(), skip_special_tokens=True)
        
        # 提取模型生成的回应部分
        response = response.split("assistant:")[-1].strip()
        
        return response
    
    def start_assessment(self):
        """开始EPDS评估"""
        # 重置评估状态
        self.answers = [None] * len(self.epds_questions)
        self.scores = [None] * len(self.epds_questions)
        self.conversation_history = []
        self.assessment_complete = False
        self.guided_interview_node.current_question_idx = 0
        self.current_node_name = "GuidedInterviewNode"
        
        # 添加欢迎信息
        welcome_message = """感谢您参与爱丁堡产后抑郁量表(EPDS)评估。

这是一份专业的筛查工具，旨在帮助了解您近期的情绪状态。接下来我会以自然对话的方式，向您询问关于过去7天内情绪体验的10个问题。

整个过程大约需要5-10分钟时间。您的回答将被保密，并仅用于评估目的。

请尽可能真实地描述您的感受，这样才能获得准确的评估结果。

准备好了吗？我们现在开始第一个问题。"""
        
        self.conversation_history.append({"role": "assistant", "content": welcome_message})
        
        # 启动第一个节点
        node = self.nodes[self.current_node_name]
        response, next_node_name = node.process()
        
        # 更新当前节点
        if next_node_name:
            self.current_node_name = next_node_name
            
        return welcome_message + "\n\n" + response
    
    def process_user_input(self, user_input):
        """
        处理用户输入，在节点之间流转
        
        参数:
            user_input: 用户输入文本
            
        返回值:
            系统回应文本
        """
        if self.assessment_complete:
            return "评估已完成。您可以查看评估报告或保存结果。"
        
        # 获取当前节点
        node = self.nodes[self.current_node_name]
        
        # 处理用户输入
        result, next_node_name = node.process(user_input)
        
        # 如果需要流转到下一个节点
        response = ""
        while next_node_name:
            # 更新当前节点
            self.current_node_name = next_node_name
            
            # 执行下一个节点
            node = self.nodes[next_node_name]
            next_result, next_node_name = node.process(result)
            
            # 更新结果和响应
            if isinstance(result, str) and isinstance(next_result, str):
                response = result + "\n\n" + next_result
            else:
                response = next_result
                
            result = next_result
        
        # 如果result不是字符串（即节点处理的结果不是直接显示的文本）
        if not isinstance(result, str):
            response = "处理完成，请继续回答问题。"
            
        return response if response else result

    def _calculate_risk_level(self, total_score):
        """根据EPDS总分确定风险级别"""
        if total_score < 10:
            return "低风险", "您的得分低于临床抑郁症状的常见阈值。这表明您目前可能没有经历显著的产后抑郁症状。"
        elif 10 <= total_score <= 12:
            return "轻度风险", "您的得分表明您可能有轻度的产后抑郁症状。建议关注自己的情绪变化，并考虑与专业人士交流。"
        elif 13 <= total_score <= 16:
            return "中度风险", "您的得分表明您可能有中度的产后抑郁症状。建议您寻求专业的心理健康支持和评估。"
        else:  # 17+
            return "高风险", "您的得分表明您可能有严重的产后抑郁症状。强烈建议您尽快咨询专业的心理健康医生进行进一步评估和支持。"
    
    def _check_suicide_risk(self):
        """检查自杀风险（基于问题10的回答）"""
        if len(self.scores) >= 10 and self.scores[9] is not None:  # 确保问题10已被回答
            q10_score = self.scores[9]  # 问题10的得分
            if q10_score >= 1:
                return True, "特别注意：您在关于自我伤害想法的问题上的回答表明需要特别关注。请考虑立即与心理健康专家交流，或联系心理健康危机热线。您的健康和安全非常重要。"
        return False, ""
    
    def get_assessment_data(self):
        """获取评估数据，用于保存或进一步分析"""
        if not self.assessment_complete:
            return {"status": "incomplete"}
        
        # 确保所有问题都已回答
        if None in self.scores or len(self.scores) < len(self.epds_questions):
            return {"status": "incomplete"}
            
        assessment_data = {
            "status": "complete",
            "total_score": sum(self.scores),
            "question_scores": [
                {
                    "question_id": self.epds_questions[i]["id"],
                    "question_text": self.epds_questions[i]["question"],
                    "selected_option": self.answers[i],
                    "score": self.scores[i]
                } for i in range(len(self.scores))
            ],
            "risk_level": self._calculate_risk_level(sum(self.scores))[0],
            "suicide_risk": self._check_suicide_risk()[0],
            "conversation_history": self.conversation_history
        }
        
        return assessment_data

    def save_assessment(self, file_path, format="json"):
        """
        保存评估结果到文件
        
        参数:
            file_path: 文件保存路径
            format: 保存格式，支持 "json" 或 "pdf"
        """
        if not self.assessment_complete:
            print("评估尚未完成，无法保存结果。")
            return False
        
        try:
            if format.lower() == "json":
                assessment_data = self.get_assessment_data()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(assessment_data, f, ensure_ascii=False, indent=2)
                print(f"评估结果已保存到: {file_path}")
                return True
            elif format.lower() == "pdf":
                return self.export_report_to_pdf(file_path)
            else:
                print(f"不支持的格式: {format}。请使用 'json' 或 'pdf'。")
                return False
        except Exception as e:
            print(f"保存评估结果时出错: {e}")
            return False
    
    def export_report_to_pdf(self, file_path):
        """
        将评估报告导出为PDF文件
        
        参数:
            file_path: PDF文件保存路径
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        except ImportError:
            print("生成PDF需要安装reportlab库，请运行: pip install reportlab")
            return False
        
        if not self.assessment_complete:
            print("评估尚未完成，无法生成PDF报告。")
            return False
        
        try:
            # 获取评估数据
            assessment_data = self.get_assessment_data()
            total_score = assessment_data["total_score"]
            risk_level = assessment_data["risk_level"]
            
            # 确定风险级别描述
            _, risk_description = self._calculate_risk_level(total_score)
            
            # 检查自杀风险
            suicide_risk, suicide_note = self._check_suicide_risk()
            
            # 创建PDF文档
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # 创建自定义样式
            title_style = styles["Heading1"]
            title_style.alignment = 1  # 居中
            
            subtitle_style = styles["Heading2"]
            subtitle_style.alignment = 1  # 居中
            
            header_style = styles["Heading3"]
            normal_style = styles["Normal"]
            
            # 新增警告样式
            warning_style = ParagraphStyle(
                "Warning",
                parent=styles["Normal"],
                textColor=colors.red,
                borderWidth=1,
                borderColor=colors.red,
                borderPadding=10,
                backColor=colors.lightgrey,
                alignment=1
            )
            
            # 创建文档内容
            content = []
            
            # 标题
            content.append(Paragraph("爱丁堡产后抑郁量表(EPDS)评估报告", title_style))
            content.append(Spacer(1, 12))
            content.append(Paragraph(f"评估日期: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            content.append(Spacer(1, 24))
            
            # 总分和风险级别
            content.append(Paragraph(f"总分: {total_score}/30", subtitle_style))
            content.append(Spacer(1, 12))
            content.append(Paragraph(f"风险级别: {risk_level}", subtitle_style))
            content.append(Spacer(1, 12))
            
            # 风险描述
            content.append(Paragraph("评估描述:", header_style))
            content.append(Paragraph(risk_description, normal_style))
            content.append(Spacer(1, 12))
            
            # 自杀风险警告（如果存在）
            if suicide_risk:
                content.append(Paragraph(suicide_note, warning_style))
                content.append(Spacer(1, 12))
            
            # 情绪波动分析
            content.append(Paragraph("情绪状态分析:", header_style))
            content.append(Spacer(1, 6))
            
            # 使用ReportGenerationNode分析情绪模式
            emotional_patterns = self.report_generation_node._analyze_emotional_patterns()
            for pattern in emotional_patterns:
                content.append(Paragraph(f"• {pattern}", normal_style))
                content.append(Spacer(1, 4))
            
            content.append(Spacer(1, 12))
            
            # 问题和回答表格
            content.append(Paragraph("回答详情:", header_style))
            content.append(Spacer(1, 6))
            
            # 表格数据
            data = [["问题", "选择的答案", "分数"]]
            for i in range(len(self.scores)):
                data.append([
                    self.epds_questions[i]["question"],
                    self.answers[i],
                    str(self.scores[i])
                ])
            
            # 创建表格
            table = Table(data, colWidths=[250, 170, 40])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(table)
            content.append(Spacer(1, 24))
            
            # 建议
            content.append(Paragraph("建议干预措施:", header_style))
            content.append(Spacer(1, 6))
            
            # 获取干预建议
            intervention_suggestions = self.report_generation_node._generate_intervention_suggestions(risk_level)
            for i, rec in enumerate(intervention_suggestions):
                content.append(Paragraph(f"{i+1}. {rec}", normal_style))
                content.append(Spacer(1, 6))
            
            content.append(Spacer(1, 12))
            
            # 免责声明
            content.append(Paragraph("重要提示:", header_style))
            content.append(Paragraph(
                "本评估报告仅用于筛查目的，不构成医疗诊断。如有任何疑虑，请咨询专业医疗人员。",
                normal_style
            ))
            
            # 构建PDF
            doc.build(content)
            print(f"PDF报告已生成: {file_path}")
            return True
        except Exception as e:
            print(f"生成PDF报告时出错: {e}")
            return False


# 命令行交互界面
def interactive_assessment(model_path, device="Ascend", device_id=0):
    """交互式EPDS评估"""
    agent = EPDSAssessmentAgent(model_path, device_target=device, device_id=device_id)
    
    print("="*80)
    print("爱丁堡产后抑郁量表(EPDS)评估系统")
    print("本系统将引导您完成10个问题的评估，以帮助了解您的情绪状态。")
    print("="*80)
    
    # 开始评估
    initial_prompt = agent.start_assessment()
    print(initial_prompt)
    
    # 主交互循环
    while not agent.assessment_complete:
        answer = input("\n您的回答: ")
        response = agent.process_user_input(answer)
        print("\n" + response)
    
    # 询问是否保存结果
    save = input("\n是否保存评估结果? (y/n): ").lower()
    if save == 'y':
        format_choice = input("请选择保存格式 (1: JSON, 2: PDF): ")
        
        if format_choice == "2":
            save_format = "pdf"
            default_filename = "epds_assessment_result.pdf"
        else:
            save_format = "json"
            default_filename = "epds_assessment_result.json"
        
        filename = input(f"请输入文件名 (默认: {default_filename}): ") or default_filename
        
        if save_format == "pdf" and not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        elif save_format == "json" and not filename.lower().endswith('.json'):
            filename += '.json'
            
        agent.save_assessment(filename, format=save_format)
    
    print("\n感谢您完成EPDS评估。如有需要，请寻求专业帮助。")


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="EPDS产后抑郁评估系统")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="微调后的模型路径")
    parser.add_argument("--device", type=str, default="Ascend", 
                        help="运行设备: Ascend, GPU, CPU")
    parser.add_argument("--device_id", type=int, default=0,
                        help="设备ID")
    
    args = parser.parse_args()
    
    interactive_assessment(args.model_path, device=args.device, device_id=args.device_id) 