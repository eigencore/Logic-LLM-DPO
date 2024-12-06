import backoff  # for exponential backoff; allows for retries
import openai
import asyncio
from typing import Any

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    """
    This function is a wrapper around the OpenAI.Completion.create function.
    
    The Completions.create function is used to generate completions for the prompt.
    """
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    """
    This function is a wrapper around the OpenAI.ChatCompletion.create function.
    
    The ChatCompletion.create function is used to generate completions for the prompt.
    """
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """
    Dispatches requests to OpenAI's API asynchronously.
    
    :param messages_list: List of messages to send to OpenAI's ChatCompletion API.
    :param model: OpenAI model to use.
    :param temperature: Temperature parameter for the model.
    :param max_tokens: Maximum number of tokens to generate.
    :param top_p: Top p to use for the model.
    :param stop_words: List of stop the model from generating tokens.
    
    return: List of completions from OpenAI's ChatCompletion API.
    """
    
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words
        )
        for x in messages_list
    ]
    
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str] 
) -> list[str]:
    """
    Dispatches requests to OpenAI's API asynchronously.
    
    :param messages_list: List of messages to send to OpenAI's Completion API.
    :param model: OpenAI model to use.
    :param temperature: Temperature parameter for the model.
    :param max_tokens: Maximum number of tokens to generate.
    :param top_p: Top p to use for the model.
    :param stop_words: List of stop the model from generating tokens.
    
    return: List of completions from OpenAI's Completion API.
    """
    
    async_responses = [
        openai.Completion.create(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_words
        )
        for x in messages_list
    ]
    
    return await asyncio.gather(*async_responses)


class OpenAIModel:
    def __init__(
        self,
        API_KEY,
        model_name,
        stop_words,
        max_new_tokens
    ) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        
    # Used by openai ChatGPT    
    def chat_generate(self, input_string, temperature=0.0):
        response = chat_completions_with_backoff(
            model = self.model_name,
            messages = [
                {"role": "user", "content": input_string}
            ],
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text
    
    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
        
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]
    
    def batch_generate(self, messages_list, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")
        
    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    
    