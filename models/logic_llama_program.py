import json
import os
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

class LLaMALogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.save_path = args.save_path
        self.model_name = args.model_name

        # Configurar LLaMA-2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Usa LlamaTokenizer para LLaMA-2
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        
        # Configuración para manejo eficiente de memoria
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto", )

        # Cuantilización opcional para hardware limitado
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     device_map="auto",
        #     quantization_config=bnb_config,  # Configuración de cuantilización
        # )

        self.model.eval()

        self.prompt_creator = {
            'FOLIO': self.prompt_folio,
            'ProntoQA': self.prompt_prontoqa,
            'ProofWriter': self.prompt_proofwriter,
            'LogicalDeduction': self.prompt_logicaldeduction,
            'AR-LSAT': self.prompt_arlsat
        }
        self.load_prompt_templates()
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        if self.dataset_name == 'AR-LSAT' and 'llama' in self.model_name.lower():
            prompt_file = f'./models/prompts/{self.dataset_name}-long.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def prompt_folio(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()})' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()})' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

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

    def logic_program_generation(self):
        # Cargar dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        for example in tqdm(raw_dataset):
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                output = self.generate(full_prompt, max_new_tokens=self.args.max_new_tokens)
                programs = [output]

                # Crear output
                output = {
                    'id': example['id'],
                    'context': example['context'],
                    'question': example['question'],
                    'answer': example['answer'],
                    'options': example['options'],
                    'raw_logic_programs': programs
                }
                outputs.append(output)
            except Exception as e:
                print(f"Error in generating logic programs for example {example['id']}: {e}")

        # Guardar resultados
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def batch_logic_program_generation(self, batch_size=10):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            try:
                for sample, full_prompt in zip(chunk, full_prompts):
                    output = self.generate(full_prompt, max_new_tokens=self.args.max_new_tokens)
                    programs = [output]
                    output = {
                        'id': sample['id'],
                        'context': sample['context'],
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'options': sample['options'],
                        'raw_logic_programs': programs
                    }
                    outputs.append(output)
            except Exception as e:
                print(f"Error in batch generation: {e}")

        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")

        file_name = f'{self.dataset_name}_{self.split}_{self.model_name.replace("/", "_")}.json'
        with open(os.path.join(self.save_path, file_name), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--model_name', type=str, default='huggyllama/llama-7b')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LLaMALogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation()
