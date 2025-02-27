import os

from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import TrainingArguments, Trainer

from Answer_Generation import load_prompts
from peft import LoraConfig, get_peft_model, TaskType


VER = 2
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

current_dir = os.path.dirname(os.path.abspath(__file__))
Model_llama = os.path.join(current_dir, 'Model/Llama/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08')
Model_deepseek = os.path.join(current_dir, 'Model/Deepseek/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562')
llama_checkpoint_path = os.path.join(current_dir, f'Model/Llama/llama_sft_checkpoints_{VER}')
llama_model_path = os.path.join(current_dir, f'Model/Llama/llama_sft_model_{VER}')
deepseek_checkpoint_path = os.path.join(current_dir, f'Model/Deepseek/deepseek_sft_checkpoints_{VER}')
deepseek_model_path = os.path.join(current_dir, f'Model/Deepseek/deepseek_sft_model_{VER}')

### NUMBER OF LAYERS TO FREEZE 
FREEZE_LAYERS = 18              # DEBERTA LARGE HAS TOTAL OF 24 LAYERS
FREEZE_EMBEDDINGS = True        # BOOLEAN TO FREEZE EMBEDDINGS

USE_PEFT = False                # Decide whether to use PEFT



# def preprocess(example, tokenizer):
#     first_sentence = "[CLS] " + example['context']
#     second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    
#     option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    
#     # Tokenize each (context, choice) pair separately
#     tokenized_example = tokenizer(
#         [first_sentence] * 5,  # Repeat the same first sentence for all choices
#         second_sentences,       # Each choice corresponds to one second sentence
#         truncation='only_first',
#         max_length=256,
#         padding='max_length',  # Ensures equal length sequences
#         return_tensors="pt"     # Returns PyTorch tensors
#     )
    
#     tokenized_example['label'] = torch.tensor(option_to_index[example['answer']], dtype=torch.long)

#     return tokenized_example
# def preprocess(example, tokenizer):
#     first_sentence = "[CLS] " + example['context']
#     second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    
#     option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    
#     # Tokenize each (context, choice) pair separately
#     tokenized_example = tokenizer(
#         [first_sentence] * 5,  # Repeat the same first sentence for all choices
#         second_sentences,       # Each choice corresponds to one second sentence
#         truncation='only_first',
#         max_length=256,
#         padding="max_length",  # Ensures equal sequence lengths
#         return_tensors=None     # FIXED: Returns lists instead of PyTorch tensors
#     )

#     tokenized_example['label'] = option_to_index[example['answer']]

#     return tokenized_example

def preprocess(example, tokenizer):

    option_to_index = {option: idx for idx, option in enumerate('ABCDE')}

    prompts = load_prompts()
    prompt = prompts.get("answer_generate_prompt", "")

    for idx in range(len(example)):
        train_ex_prompt = example.loc[idx,'prompt']
        train_ex_multiple_choice_df = example.loc[idx,['A','B','C','D','E']]
        train_ex_multiple_choice = ""
        for index, i in zip(list(train_ex_multiple_choice_df.index), list(train_ex_multiple_choice_df.values)):
            train_ex_multiple_choice += index + ' : ' + i + '\n'
    
        final_prompt = prompt.replace("{Scientific_Problem}", train_ex_prompt)\
                            .replace("{Multiple_Choice}", train_ex_multiple_choice)
        
        example.loc[idx,'final_prompt'] = final_prompt

    example = example.drop(columns = [i for i in example.columns() if i not in ['final_prompt','answer']])

    tokenized_example = example.map(lambda x: tokenizer(x['final_prompt']).to(model.device))
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example



def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}
    


# @dataclass
# class DataCollatorForMultipleChoice:
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
    
#     def __call__(self, features):
#         label_name = 'label' if 'label' in features[0].keys() else 'labels'
#         labels = [feature.pop(label_name) for feature in features]
#         batch_size = len(features)
#         num_choices = len(features[0]['input_ids'])
#         flattened_features = [
#             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
#         ]
#         flattened_features = sum(flattened_features, [])
        
#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors='pt',
#         )
#         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
#         batch['labels'] = torch.tensor(labels, dtype=torch.int64)
#         return batch
@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])  # Ensure correct shape
        
        # Flattening choice inputs
        flattened_features = [
            {k: v[i] for k, v in feature.items()} for feature in features for i in range(num_choices)
        ]

        # Tokenizer padding
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",  # Ensures PyTorch compatibility
        )

        # Reshape to match [batch_size, num_choices, seq_len]
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.long)

        return batch
    
    
    
    
###### 코드 실행 ########
if __name__ == '__main__':
    
    ### 1. Data Load
    df_valid = pd.read_csv(os.path.join(current_dir, 'Data/SFT/train_with_context2.csv'))
    df_train = pd.read_csv(os.path.join(current_dir, 'Data/SFT/all_12_with_context2.csv'))
    df_train = df_train.drop(columns="source")
    df_train = df_train.fillna('').sample(1024)     # Training with 1024 samples

    dataset_valid = Dataset.from_pandas(df_valid)
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.remove_columns(["__index_level_0__"])
    
    
    ### 2. Training
    # (1) Llama
    
    
    # (2) deepseek
    tokenizer = AutoTokenizer.from_pretrained(Model_deepseek)
    # tokenized_dataset_valid = dataset_valid.map(preprocess, fn_kwargs={'tokenizer': tokenizer}, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    # tokenized_dataset = dataset.map(preprocess, fn_kwargs={'tokenizer': tokenizer}, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    
    tokenized_dataset_valid = dataset_valid.map(
        preprocess, 
        fn_kwargs={'tokenizer': tokenizer}, 
        remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']
    )

    tokenized_dataset = dataset.map(
        preprocess, 
        fn_kwargs={'tokenizer': tokenizer}, 
        remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']
    )
    model = AutoModelForCausalLM.from_pretrained(Model_deepseek)
    
    print('불러오기 성공')
    if USE_PEFT:
        peft_config = LoraConfig(
            r=8, lora_alpha=4, task_type=TaskType.SEQ_CLS, lora_dropout=0.1, 
            bias="none", inference_mode=False, 
            target_modules=["query_proj", "value_proj"],
            modules_to_save=['classifier','pooler'],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        
    # if FREEZE_EMBEDDINGS:
    #     print('Freezing embeddings.')
    #     for param in model.deberta.embeddings.parameters():
    #         param.requires_grad = False
            
    # if FREEZE_LAYERS>0:
    #     print(f'Freezing {FREEZE_LAYERS} layers.')
    #     for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
    #         for param in layer.parameters():
    #             param.requires_grad = False

                
    print('직전까지 성공')
    training_args = TrainingArguments(
        warmup_ratio=0.1, 
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        report_to='none',
        output_dir = deepseek_checkpoint_path,
        overwrite_output_dir=True,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        load_best_model_at_end=False,
        metric_for_best_model='map@3',
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        save_total_limit=2,
    )

    
    print('training 시작')
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics = compute_metrics,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )


    trainer.train()
    trainer.save_model(deepseek_model_path)

    