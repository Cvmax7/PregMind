import os
import sys
import json
import time
import argparse

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入EPDS评估智能体
from Agent_EPDS.EPDS_Assessment_Agent import EPDSAssessmentAgent

# 导入心理健康支持智能体
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Agent_RLHF"))
from Agent_RLHF.MentalHealth_Agent import MaternalMentalHealthAgent

class MultiAgentSystem:
    """
    多智能体协同系统 - 整合EPDS评估和心理健康支持功能
    """
    
    def __init__(self, epds_model_path, mental_health_model_path, device="cuda", device_id=0):
        """
        初始化多智能体系统
        
        参数:
            epds_model_path: EPDS评估智能体模型路径
            mental_health_model_path: 心理健康支持智能体模型路径
            device: 推理设备
            device_id: 设备ID
        """
        print("=" * 80)
        print("初始化多智能体孕产妇心理健康支持系统")
        print("=" * 80)
        
        # 根据设备类型选择MindSpore设备
        ms_device = "Ascend" if "cuda" in device.lower() else device
        
        # 初始化Agent1: EPDS评估智能体
        print("\n[1/2] 正在初始化EPDS评估智能体...")
        self.agent1 = EPDSAssessmentAgent(
            model_path=epds_model_path,
            device_target=ms_device,
            device_id=device_id
        )
        
        # 初始化Agent2: 心理健康支持智能体
        print("\n[2/2] 正在初始化心理健康支持智能体...")
        self.agent2 = MaternalMentalHealthAgent(
            model_path=mental_health_model_path,
            device=device,
            thought_format=("<思考>", "</思考>")
        )
        
        # 当前活动智能体
        self.active_agent = "agent1"  # 从EPDS评估开始
        self.transfer_complete = False  # 是否已完成智能体转换
        
        print("\n系统初始化完成，准备开始交互。")
        print("=" * 80)
    
    def _format_agent2_context(self, transfer_context):
        """
        格式化传递给Agent2的上下文
        
        参数:
            transfer_context: Agent1传递的上下文信息
            
        返回值:
            格式化后的问题文本
        """
        # 提取关键信息
        assessment = transfer_context["assessment_result"]
        total_score = assessment["total_score"]
        risk_level = assessment["risk_level"]
        emotional_patterns = assessment["emotional_patterns"]
        suicide_risk = assessment["suicide_risk"]
        user_expressions = transfer_context["user_expressions"]
        assessment_time = transfer_context["assessment_time"]
        
        # 构建提示词
        prompt = f"类型 问题描述#EPDS量表评估结果显示，来访者得分{total_score}分，风险级别为{risk_level}。"
        
        # 添加情绪模式描述
        if emotional_patterns:
            prompt += f"情绪状态分析显示："
            for pattern in emotional_patterns:
                prompt += f"{pattern} "
        
        # 添加自杀风险警告
        if suicide_risk:
            prompt += "特别需要注意的是，存在自伤风险。 "
        
        # 添加用户关键表述
        if user_expressions:
            prompt += "来访者曾表达："
            for expr in user_expressions:
                prompt += f"\"{expr}\" "
        
        # 添加关键词标记
        prompt += f"关键词#产后抑郁, EPDS评估, {risk_level}, "
        
        if "焦虑" in "".join(emotional_patterns):
            prompt += "焦虑, "
        if "睡眠" in "".join(emotional_patterns):
            prompt += "睡眠问题, "
        if suicide_risk:
            prompt += "自伤风险, "
        
        # 去除最后的逗号和空格
        prompt = prompt.rstrip(", ")
        
        return prompt
    
    def start_session(self):
        """开始多智能体协同会话"""
        # 启动Agent1进行EPDS评估
        initial_prompt = self.agent1.start_assessment()
        print(initial_prompt)
        
        # 主交互循环
        while True:
            # 获取用户输入
            user_input = input("\n您的回答: ")
            
            # 如果用户输入"退出"，则结束会话
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("\n会话已结束。")
                break
            
            # 根据当前活动智能体处理用户输入
            if self.active_agent == "agent1":
                # 使用Agent1处理用户输入
                response = self.agent1.process_user_input(user_input)
                print(f"\n{response}")
                
                # 检查是否需要转接到Agent2
                if self.agent1.assessment_complete and self.agent1.should_transfer is True:
                    # 获取转接上下文
                    transfer_context = self.agent1.transfer_context
                    
                    # 格式化上下文为Agent2可接受的格式
                    formatted_context = self._format_agent2_context(transfer_context)
                    
                    # 切换到Agent2
                    self.active_agent = "agent2"
                    self.transfer_complete = True
                    
                    # 使用Agent2生成初始响应
                    print("\n正在连接心理健康支持顾问，请稍候...\n")
                    time.sleep(1)  # 模拟连接延迟
                    
                    # 生成Agent2的初始响应
                    initial_response = self.agent2.generate_response(
                        formatted_context,
                        include_thinking=True,
                        show_thinking=False
                    )
                    
                    print("-" * 80)
                    print("您已连接到心理健康支持顾问")
                    print("-" * 80)
                    print(f"\n{initial_response}")
                    
                # 如果用户明确拒绝转接或评估显示风险较低
                elif self.agent1.assessment_complete and self.agent1.should_transfer is False:
                    print("\n会话已完成。如有需要，请随时寻求专业帮助。")
                    break
            
            elif self.active_agent == "agent2":
                # 使用Agent2处理用户输入
                response = self.agent2.generate_response(
                    user_input,
                    include_thinking=True,
                    show_thinking=False
                )
                
                print(f"\n{response}")
    
    def save_session_data(self, file_path):
        """
        保存会话数据
        
        参数:
            file_path: 会话数据保存路径
        """
        session_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "transfer_happened": self.transfer_complete
        }
        
        # 添加Agent1的评估数据
        if self.agent1.assessment_complete:
            session_data["epds_assessment"] = self.agent1.get_assessment_data()
        
        # 添加会话历史
        session_data["conversation_history"] = self.agent1.conversation_history
        
        # 保存到文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            print(f"会话数据已保存到: {file_path}")
            return True
        except Exception as e:
            print(f"保存会话数据时出错: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="孕心导航")
    parser.add_argument("--epds_model", type=str, required=True, 
                        help="EPDS评估智能体模型路径")
    parser.add_argument("--mental_health_model", type=str, required=True,
                        help="心理健康支持智能体模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备: cuda, cpu")
    parser.add_argument("--device_id", type=int, default=0,
                        help="设备ID")
    parser.add_argument("--save_path", type=str, default="session_data.json",
                        help="会话数据保存路径")
    
    args = parser.parse_args()
    
    # 初始化多智能体系统
    system = MultiAgentSystem(
        epds_model_path=args.epds_model,
        mental_health_model_path=args.mental_health_model,
        device=args.device,
        device_id=args.device_id
    )
    
    try:
        # 开始会话
        system.start_session()
    except KeyboardInterrupt:
        print("\n用户中断，会话结束。")
    except Exception as e:
        print(f"\n会话过程中出现错误: {e}")
    finally:
        # 询问是否保存会话数据
        save = input("\n是否保存会话数据? (y/n): ").lower()
        if save == 'y':
            system.save_session_data(args.save_path)
        
        print("\n感谢使用孕心导航。")


if __name__ == "__main__":
    main() 