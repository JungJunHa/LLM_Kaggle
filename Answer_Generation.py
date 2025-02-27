import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Model
from openai import OpenAI   # 1) Chatgpt - o3 mini 2) Llama 3.2(8B) 3) deepseek r1-distilled Qwen(7B)
import ollama               # 3) deepseek r1

import yaml
from dotenv import load_dotenv
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch


# 환경변수 로드 (.env 파일에 OPENAI_API_KEY, ANTHROPIC_API_KEY, PATH_DATA, PATH_ASSET 정의)
load_dotenv()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(current_dir, 'Data')
PATH_ASSET = os.path.join(current_dir, 'Asset')
PATH_MODEL_LLAMA = ''
PATH_MODEL_DEEPSEEK = ''
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")




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
    
    ### HuggingFace
    
    # 1.Llama 모델 다운로드
    llama_path = os.path.join(current_dir, 'Model/Llama')
    
    if not os.path.exists(llama_path):
        print("Llama 모델을 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        cache_dir=llama_path
        )

        model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        cache_dir=llama_path,
        device_map="auto",
        ).to(device)
        
        print(f"Model and tokenizer downloaded in: {llama_path}")
    else:
        print("Llama 모델이 이미 존재합니다. 다운로드를 건너뜁니다.")
    
    
    # 2. Deepseek-R1 모델 다운로드
    deepseek_path = os.path.join(current_dir, 'Model/Deepseek')
    
    if not os.path.exists(deepseek_path):
        print("Deepseek-R1 모델을 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        cache_dir=deepseek_path
        )

        model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        cache_dir=deepseek_path,
        device_map="auto",
        )
        
        print(f"Model and tokenizer downloaded in: {deepseek_path}")
    else:
        print("Deepseek-R1 모델이 이미 존재합니다. 다운로드를 건너뜁니다.")





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
                model="o3-mini-2025-01-31",
                messages=[{"role": "user", "content": final_prompt}]
                )
            
            response_message = response.choices[0].message.content
            return response_message
        
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    # 2. Llama
    elif model.lower() in ["llama", "llama3", "llama3.2"]:
        try:
            llama_path = os.path.join(current_dir, 'Model/Llama/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08')

            tokenizer = AutoTokenizer.from_pretrained(llama_path)
            model = AutoModelForCausalLM.from_pretrained(llama_path, device_map="auto").to(device)
            
            # Tokenize input prompt
            inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
                        
            # Generate response
            output = model.generate(**inputs,
                                    max_length=600,
                                    temperature=0.7,
                                    top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id
                                    )
            
            # Decode and return response
            return tokenizer.decode(output[0], skip_special_tokens=True)
        
        except Exception as e:
            return f"Error running Llama3 : {e}"
    
    # 3. DeepSeek-R1
    elif model.lower() in ["deepseek", "deepseek-r1"]:
        try:
            deepseek_path = os.path.join(current_dir, 'Model/Deepseek/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562')

            tokenizer = AutoTokenizer.from_pretrained(deepseek_path)
            model = AutoModelForCausalLM.from_pretrained(deepseek_path, device_map="auto").to(device)
            
            # Tokenize input prompt
            inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
                        
            # Generate response
            output = model.generate(**inputs,
                                    max_length=600,
                                    temperature=0.7,
                                    top_p=0.9,
                                    pad_token_id=tokenizer.eos_token_id
                                    )
            
            # Decode and return response
            return tokenizer.decode(output[0], skip_special_tokens=True)

            
        except Exception as e:
            return f"Error running Deepseek : {e}"
    
    else:
        return "Unsupported model."




###### 코드 실행 ########
if __name__ == '__main__':
    
    login_hugginface = True
    
    # login
    if login_hugginface:
        login(token = HUGGINGFACE_TOKEN)
        
    download_model()
    
    
    # yaml prompt 가져오기
    prompts = load_prompts()
    prompt = prompts.get("answer_generate_prompt", "")
    
    # train 데이터 가져오기
    train_path = os.path.join(PATH_DATA, 'train.csv')
    train_df = pd.read_csv(train_path, encoding = 'utf-8')
    
    train_ex_prompt = train_df.loc[0,'prompt']
    train_ex_multiple_choice_df = train_df.loc[0,['A','B','C','D','E']]
    train_ex_multiple_choice = ""
    for index, i in zip(list(train_ex_multiple_choice_df.index), list(train_ex_multiple_choice_df.values)):
        train_ex_multiple_choice += index + ' : ' + i + '\n'
        
    
    model_list = ['llama','deepseek']    
    for index, model in enumerate(model_list):
        feedback_res = feedback_generation(model = model, prompt = prompt,
                                            Scientific_Problem = train_ex_prompt,
                                            Multiple_Choice = train_ex_multiple_choice)
        
        print(str(index+1) + '.' + model + ' : ' + feedback_res + '\n\n')
