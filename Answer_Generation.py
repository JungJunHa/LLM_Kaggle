import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Model
from openai import OpenAI   # 1) Chatgpt - o3 mini
import anthropic            # 2) anthropic - claude
import ollama               # 3) deepseek r1

import yaml
from dotenv import load_dotenv



# 환경변수 로드 (.env 파일에 OPENAI_API_KEY, ANTHROPIC_API_KEY, PATH_DATA, PATH_ASSET 정의)
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
PATH_DATA = './Data'
PATH_ASSET = './Asset'

anthropic_client = anthropic.Client(api_key=anthropic_api_key)




def load_prompts(file_path=PATH_ASSET + "prompt.yaml"):
    """
    YAML 파일로부터 프롬프트들을 로드합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        raise RuntimeError(f"Error loading prompt file: {e}")

    
def feedback_generation(model, prompt, env_info, target_info):
    """
    지정한 모델과 프롬프트를 사용하여 피드백을 생성합니다.
    """

    final_prompt = prompt.replace("{Research_Environment}", env_info)\
                        .replace("{Target}", target_info)

    # 1. GPT-4o
    if model.lower() in ["gpt", "openai"]:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.7
            )
            response_message = response.choices[0].message.content
            return response_message
        
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    # 2. Claude
    elif model.lower() in ["anthropic", "claude"]:
        try:
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                messages=[{"role": "user", "content": final_prompt}]
            )
            return message.content, final_prompt
        
        except Exception as e:
            return f"Error calling Anthropic API: {e}"
    
    # 3. DeepSeek-R1
    elif model.lower() in ["deepseek", "r1"]:
        print('hah')
    
    else:
        return "Unsupported model."



