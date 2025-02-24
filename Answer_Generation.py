import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Model
from openai import OpenAI   # 1) Chatgpt - o3 mini 2) Llama
import ollama               # 3) deepseek r1

import yaml
from dotenv import load_dotenv
import transformers
from transformers import pipeline
import subprocess
import torch


# 환경변수 로드 (.env 파일에 OPENAI_API_KEY, ANTHROPIC_API_KEY, PATH_DATA, PATH_ASSET 정의)
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(current_dir, 'Data')
PATH_ASSET = os.path.join(current_dir, 'Asset')
PATH_MODEL_LLAMA = ''
PATH_MODEL_DEEPSEEK = ''




def load_prompts(file_path=PATH_ASSET + "/prompt.yaml"):
    """
    YAML 파일로부터 프롬프트들을 로드합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        return prompts
    except Exception as e:
        raise RuntimeError(f"Error loading prompt file: {e}")


def download_model():
    """
    Llama와 Deepseek-R1 모델을 다운받아 활용하되, 해당 repository에 없으면 pull해서 다운로드하기
    """
    
    # 현재 파일이 있는 디렉토리 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ollama server 열기
    try : 
        subprocess.run(["ollama", "serve"], check=True)
    except Exception as e:
        print('Server is running already')
    
    # Llama 모델 다운로드
    llama_path = os.path.join(current_dir, 'Llama')
    if not os.path.exists(llama_path):
        print("Llama 모델을 다운로드 중...")
        subprocess.run(["ollama", "serve"], check=True)
        subprocess.run(["ollama", "pull", "llama3.2"], check=True)
        os.makedirs(llama_path, exist_ok=True)
    else:
        print("Llama 모델이 이미 존재합니다. 다운로드를 건너뜁니다.")
    
    # Deepseek-R1 모델 다운로드
    deepseek_path = os.path.join(current_dir, 'Deepseek')
    if not os.path.exists(deepseek_path):
        print("Deepseek-R1 모델을 다운로드 중...")
        subprocess.run(["ollama", "serve"], check=True)
        subprocess.run(["ollama", "pull", "deepseek-r1"], check=True)
        os.makedirs(deepseek_path, exist_ok=True)
    else:
        print("Deepseek-R1 모델이 이미 존재합니다. 다운로드를 건너뜁니다.")



class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids(""),
        ]
  
    def get_response(
          self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response
    



def feedback_generation(model, prompt, Scientific_Problem, Multiple_Choice):
    """
    지정한 모델과 프롬프트를 사용하여 피드백을 생성합니다.
    """

    final_prompt = prompt.replace("{Scientific_Problem}", Scientific_Problem)\
                        .replace("{Multiple_Choice}", Multiple_Choice)

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
    
    # 2. Llama
    elif model.lower() in ["llama", "llama3", "llama3.2"]:
        try:
            template_llama = [
                {"role": "user", "content": final_prompt},
            ]
            llama_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")
            message = llama_pipeline(template_llama)
            
            return message['generated_text']['content']
        
        except Exception as e:
            return f"Error running Llama3 : {e}"
    
    # 3. DeepSeek-R1
    elif model.lower() in ["deepseek", "deepseek-r1"]:
        try:
            template_deepseek = [
                {"role": "user", "content": final_prompt},
            ]
            deepseek_pipeline = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            message = deepseek_pipeline(template_deepseek)
            
            return message['generated_text']['content']

        except Exception as e:
            return f"Error running Deepseek : {e}"
    
    else:
        return "Unsupported model."




###### 코드 실행 ########
if __name__ == '__main__':
    
    # yaml prompt 가져오기
    prompt = load_prompts()
    
    # train 데이터 가져오기
    train_path = os.path.join(PATH_DATA, 'train.csv')
    train_df = pd.read_csv(train_path, encoding = 'utf-8')
    
    train_ex_prompt = train_df.loc[0,'prompt']
    train_ex_multiple_choice_df = train_df.loc[0,['A','B','C','D','E']]
    train_ex_multiple_choice = ""
    for index, i in zip(list(train_ex_multiple_choice_df.index), list(train_ex_multiple_choice_df.values)):
        train_ex_multiple_choice += index + ' : ' + i + '\n'
        
    
    model_list = ['gpt','llama','deepseek']    
    for index, model in enumerate(model_list):
        feedback_res = feedback_generation(model = model, prompt = train_ex_prompt,
                                            Scientific_Problem = train_ex_prompt,
                                            Multiple_Choice = train_ex_multiple_choice)
        
        print(str(index+1) + '.' + model + ' : ' + feedback_res + '\n\n')
