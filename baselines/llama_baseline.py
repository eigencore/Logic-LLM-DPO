import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

class LLaMAReasoningGraphBaseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode
        
        # Configurar modelo LLaMA
        self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()

        self.label_phrase = 'The correct option is:'

    def prompt_LSAT(self, in_context_example, test_example):
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        return full_prompt
    
    def load_in_context_examples(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"], max_length=len(inputs["input_ids"][0]) + max_new_tokens, do_sample=True, top_p=0.95
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example['question']

            # create prompt
            full_prompt = self.prompt_LSAT(in_context_examples, example)
            output = self.generate(full_prompt, max_new_tokens=self.args.max_new_tokens)
            
            # get the answer
            label_phrase = self.label_phrase
            generated_answer = output.split(label_phrase)[-1].strip()
            generated_reasoning = output.split(label_phrase)[0].strip()

            # create output
            output = {'id': example['id'], 
                      'question': question, 
                      'answer': example['answer'], 
                      'predicted_reasoning': generated_reasoning,
                      'predicted_answer': generated_answer}
            outputs.append(output)

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
            
    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()

        outputs = []
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            for sample in chunk:
                full_prompt = self.prompt_LSAT(in_context_examples, sample)
                try:
                    output = self.generate(full_prompt, max_new_tokens=self.args.max_new_tokens)
                    dict_output = self.update_answer(sample, output)
                    outputs.append(dict_output)
                except:
                    print(f"Error in generating example: {sample['id']}")
        
        file_name = f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name.replace("/", "_")}.json'
        os.makedirs(self.save_path, exist_ok=True)

        with open(os.path.join(self.save_path,file_name), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
    def update_answer(self, sample, output):
        label_phrase = self.label_phrase
        generated_answer = output.split(label_phrase)[-1].strip()
        generated_reasoning = output.split(label_phrase)[0].strip()
        dict_output = {'id': sample['id'], 
                       'question': sample['question'], 
                       'answer': sample['answer'], 
                       'predicted_reasoning': generated_reasoning,
                       'predicted_answer': generated_answer}
        return dict_output
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--model_name', type=str, default="huggyllama/llama-7b")
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    llama_reasoning = LLaMAReasoningGraphBaseline(args)
    llama_reasoning.batch_reasoning_graph_generation(batch_size=10)
