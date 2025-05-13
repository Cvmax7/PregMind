import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse
import json
import re

class MaternalMentalHealthAgent:
    """
    孕产妇心理健康支持智能体 - 基于RLHF训练的LLaMA3模型，融合思维链(CoT)和认知行为疗法(CBT)技术
    """
    def __init__(self, model_path, device="cuda", thought_format=("<思考>", "</思考>")):
        """
        初始化孕产妇心理健康支持智能体
        
        参数:
            model_path: RLHF训练后的模型路径
            device: 推理设备
            thought_format: 思考标记的元组 (开始标记, 结束标记)
        """
        print(f"正在加载模型，路径: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device
        )
        
        self.thought_start, self.thought_end = thought_format
        self.device = device
        
        # 配置
        self.generation_config = GenerationConfig(
            temperature=0.75,  # 略微提高温度增加情感表达多样性
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.15,  # 提高重复惩罚，避免常见安慰语重复
            max_new_tokens=2048,
            do_sample=True,
        )
        
        # 孕产妇常见压力源及情绪问题类别
        self.maternal_stressors = [
            "生理变化", "分娩焦虑", "产后恢复", "身体不适", "激素变化",
            "育儿压力", "喂养问题", "照顾新生儿", "睡眠剥夺", "婴儿健康",
            "伴侣关系", "家庭支持", "角色转变", "职业规划", "经济压力",
            "社会期望", "自我认同", "产后抑郁", "产后焦虑", "情绪波动"
        ]
        
        # CBT技术核心组件
        self.cbt_components = {
            "思想识别": "帮助识别负面自动思想和认知扭曲",
            "思想质疑": "挑战非理性信念和负面思考模式",
            "行为激活": "鼓励积极行为和活动参与",
            "应对技能": "提供应对技巧和解决问题的策略",
            "情绪调节": "教授情绪调节和压力管理技巧"
        }
    
    def identify_stressors(self, question):
        """
        识别问题中可能的孕产妇压力源
        
        参数:
            question: 用户问题
        返回:
            相关压力源列表
        """
        identified = []
        for stressor in self.maternal_stressors:
            if stressor in question:
                identified.append(stressor)
            
        # 如果没有明确匹配，尝试进行语义关联
        if not identified:
            if any(word in question for word in ["疼", "痛", "不舒服", "身体", "恢复"]):
                identified.append("生理变化")
                identified.append("产后恢复")
            
            if any(word in question for word in ["焦虑", "担心", "害怕", "紧张", "恐惧", "生产"]):
                identified.append("分娩焦虑")
                identified.append("产后焦虑")
                
            if any(word in question for word in ["宝宝", "婴儿", "孩子", "照顾", "喂养", "母乳", "哺乳"]):
                identified.append("育儿压力")
                identified.append("喂养问题")
                
            if any(word in question for word in ["伴侣", "丈夫", "老公", "家庭", "婆婆", "关系"]):
                identified.append("伴侣关系")
                identified.append("家庭支持")
                
            if any(word in question for word in ["抑郁", "难过", "伤心", "失落", "自责", "情绪低落"]):
                identified.append("产后抑郁")
                identified.append("情绪波动")
                
            if any(word in question for word in ["睡眠", "睡不着", "休息", "疲惫", "累"]):
                identified.append("睡眠剥夺")
        
        return identified[:3]  # 返回最相关的前三个压力源
        
    def apply_cbt_technique(self, stressors):
        """
        根据识别到的压力源选择适当的CBT技术
        
        参数:
            stressors: 识别到的压力源列表
        返回:
            推荐的CBT技术和应用方向
        """
        cbt_response = {
            "techniques": [],
            "guidance": ""
        }
        
        if "产后抑郁" in stressors or "情绪波动" in stressors or "产后焦虑" in stressors:
            cbt_response["techniques"].append("思想识别")
            cbt_response["techniques"].append("思想质疑")
            cbt_response["guidance"] += "引导识别负面自动思想，协助质疑不合理信念。"
            
        if "育儿压力" in stressors or "喂养问题" in stressors or "睡眠剥夺" in stressors:
            cbt_response["techniques"].append("行为激活")
            cbt_response["techniques"].append("应对技能")
            cbt_response["guidance"] += "建议具体可行的应对策略，促进积极行为改变。"
            
        if "伴侣关系" in stressors or "家庭支持" in stressors or "社会期望" in stressors:
            cbt_response["techniques"].append("应对技能")
            cbt_response["guidance"] += "提供沟通技巧和问题解决策略，增强社会支持。"
            
        if "分娩焦虑" in stressors or "生理变化" in stressors or "身体不适" in stressors:
            cbt_response["techniques"].append("情绪调节")
            cbt_response["guidance"] += "教授放松技巧和接纳策略，改善身体感知。"
            
        # 如果没有匹配到具体策略，提供通用CBT思路
        if not cbt_response["techniques"]:
            cbt_response["techniques"] = ["思想识别", "情绪调节", "应对技能"]
            cbt_response["guidance"] = "从认知、情绪和行为三方面提供综合支持。"
            
        return cbt_response
    
    def format_cot_prompt(self, question, include_thinking=True):
        """
        格式化具有思维链的提示词，并整合CBT技术
        
        参数:
            question: 用户问题
            include_thinking: 是否包含思考指令
        返回:
            格式化后的消息列表
        """
        # 识别潜在压力源
        stressors = self.identify_stressors(question)
        
        # 应用CBT技术
        cbt_approach = self.apply_cbt_technique(stressors)
        
        # 提取关键词（如果有）
        keywords = []
        description = question
        
        if "关键词#" in question:
            parts = question.split("关键词#")
            if len(parts) > 1:
                description = parts[0].replace("类型 问题描述#", "").strip()
                keywords_text = parts[1].strip()
                keywords = [k.strip() for k in keywords_text.split("，")]
        
        # 构建系统提示信息
        system_message = (
            "你是一位专业的孕产妇心理健康顾问，拥有丰富的产科心理咨询经验和深厚的专业知识。"
            "你的目标是为孕产妇提供心理支持、情感共情和实用建议，缓解她们在怀孕和产后阶段面临的各种压力和情绪困扰。"
        )
        
        # 添加有关当前用户问题的针对性指导
        if stressors:
            system_message += f"\n\n当前用户似乎面临以下压力源：{', '.join(stressors)}。"
        
        # 添加CBT应用指导
        if cbt_approach["techniques"]:
            system_message += f"\n请应用以下认知行为疗法(CBT)技术：{', '.join(cbt_approach['techniques'])}。{cbt_approach['guidance']}"
        
        # 添加思维链指导
        if include_thinking:
            system_message += f"\n\n请先使用{self.thought_start}和{self.thought_end}标签进行系统深入思考，然后再给出回答。思考过程应包含："
            system_message += "\n1. 问题分析：理解孕产妇表达的具体困扰和潜在需求"
            system_message += "\n2. 心理评估：专业角度评估可能的心理状态和风险程度"
            system_message += "\n3. CBT技术应用：选择合适的认知行为疗法技术和实施方向"
            system_message += "\n4. 干预策略：制定个性化支持方案和具体建议"
            system_message += "\n5. 沟通方式：确定温和、专业且共情的回应语气和表达方式"
        
        # 添加回答指导
        system_message += "\n\n你的回应应当：具备专业性、体现深度共情、提供实用建议、语气温和支持、避免医疗诊断、鼓励积极行动。"
        
        # 格式化消息
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        return messages
    
    def add_structured_thinking(self, response):
        """
        为响应添加结构化思考元素（如果不存在）
        
        参数:
            response: 原始响应文本
        返回:
            添加结构化思考后的文本
        """
        if self.thought_start not in response:
            structured_thinking = (
                f"{self.thought_start}\n"
                "1. 问题分析：理解这位孕产妇表达的具体困扰和潜在需求...\n"
                "2. 心理评估：从专业角度评估可能的心理状态和风险程度...\n"
                "3. CBT技术应用：选择合适的认知行为疗法技术...\n"
                "4. 干预策略：设计个性化支持方案和具体建议...\n"
                "5. 沟通方式：采用温和、专业且共情的语气...\n"
                f"{self.thought_end}\n\n"
            )
            return structured_thinking + response
        return response
    
    def generate_response(self, question, include_thinking=True, show_thinking=False):
        """
        使用思维链技术和CBT方法生成回应
        
        参数:
            question: 用户问题
            include_thinking: 是否在提示中包含思考指令
            show_thinking: 是否在最终输出中显示思考过程
        返回:
            生成的回应文本
        """
        messages = self.format_cot_prompt(question, include_thinking)
        
        # 获取输入文本
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 如果需要，在输入中插入思考结构
        if include_thinking and "assistant" in input_text.lower() and self.thought_start not in input_text:
            # 找到助手部分
            start_idx = input_text.lower().rfind("<|start_header_id|>assistant<|end_header_id|>\n\n")
            if start_idx != -1:
                # 在助手头部之后插入思考结构
                end_idx = start_idx + len("<|start_header_id|>assistant<|end_header_id|>\n\n")
                structured_thinking = (
                    f"{self.thought_start}\n"
                    "请进行系统思考...\n"
                    f"{self.thought_end}\n\n"
                )
                input_text = input_text[:end_idx] + structured_thinking + input_text[end_idx:]
        
        # 生成回应
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            generation_config=self.generation_config,
            attention_mask=inputs["attention_mask"],
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手部分
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # 移除结束标记
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
        
        # 处理思考部分
        if not show_thinking and self.thought_start in response and self.thought_end in response:
            thinking_start = response.find(self.thought_start)
            thinking_end = response.find(self.thought_end) + len(self.thought_end)
            
            # 仅在找到两个标记时移除
            if thinking_start != -1 and thinking_end != -1:
                response = response[:thinking_start] + response[thinking_end:].lstrip()
                
                # 强化CBT应用，确保回应包含认知行为疗法的关键元素
                if not any(cbt_elem in response for cbt_elem in ["认知模式", "自动思想", "行为激活", "情绪调节", "认知重构"]):
                    # 提取识别到的压力源和应用的CBT技术
                    identified_stressors = self.identify_stressors(question)
                    cbt_approach = self.apply_cbt_technique(identified_stressors)
                    
                    # 分析响应文本结构，寻找最佳插入点
                    response_paragraphs = response.split('\n\n')
                    if len(response_paragraphs) >= 2:
                        # 在第二段后插入CBT应用提示
                        cbt_paragraph = f"让我们运用认知行为疗法的方法来应对这种情况。首先，请留意你的想法和情绪之间的联系，然后我们可以一起探索更加平衡的思考方式和有效的应对策略。"
                        response_paragraphs.insert(2, cbt_paragraph)
                        response = '\n\n'.join(response_paragraphs)
        
        return response.strip()
    
    def load_data(self, file_path):
        """
        加载样例问题数据集
        
        参数:
            file_path: 数据集文件路径
        返回:
            加载的数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

def main():
    parser = argparse.ArgumentParser(description="孕产妇心理健康支持智能体 - 基于LLaMA3和认知行为疗法(CBT)")
    parser.add_argument("--model_path", type=str, default="output/Meta-Llama-3-8B-Instruct-psyQA-RLHG",
                        help="RLHF训练后的模型路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--question", type=str, default=None,
                        help="自定义问题（不提供则使用样例）")
    parser.add_argument("--dataset_path", type=str, default="dataset/PsyQA.json",
                        help="样例问题数据集路径")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="从数据集选择的样例索引")
    parser.add_argument("--show_thinking", action="store_true",
                        help="是否在输出中显示思考过程")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(os.getcwd(), args.model_path)
        
    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.join(os.getcwd(), args.dataset_path)
    
    # 初始化智能体
    agent = MaternalMentalHealthAgent(
        model_path=args.model_path,
        device=args.device,
        thought_format=("<思考>", "</思考>")  # 使用中文思考标记提高可读性
    )
    
    # 获取问题（从参数或数据集）
    question = args.question
    if question is None:
        try:
            data = agent.load_data(args.dataset_path)
            if args.sample_index < len(data):
                sample = data[args.sample_index]
                question = sample["instruction"]
                print(f"使用样例问题: {question}")
            else:
                raise IndexError(f"样例索引 {args.sample_index} 超出数据集范围，数据集共有 {len(data)} 条记录")
        except Exception as e:
            print(f"加载数据集出错: {e}")
            question = "我怀孕7个月了，最近总是担心分娩时会出现各种问题，晚上睡不好，白天也总是紧张。感觉自己的情绪波动很大，丈夫经常不理解我。我该怎么缓解这种焦虑状态？"
            print(f"使用默认问题: {question}")
    
    # 生成回应
    print("\n正在生成回应，使用思维链(CoT)和认知行为疗法(CBT)技术...\n")
    response = agent.generate_response(question, include_thinking=True, show_thinking=args.show_thinking)
    
    print("-" * 80)
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main() 