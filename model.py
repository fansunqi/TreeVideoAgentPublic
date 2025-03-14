import openai
from openai import OpenAI
import pdb
from pprint import pprint
import time
import json
from util import get_from_cache, save_to_cache

LOG_MAX_LENGTH = 300

# post_process_fn 的一个示例
def identity(res):
    return res


def get_model(args):
    # 选择不同模型，如 GPT, TogetherAI
    model_name, temperature = args.model, args.temperature
    base_url = args.openai_proxy if hasattr(args, 'openai_proxy') else None
    print('base_url: ', base_url)
    
    if 'gpt' in model_name:
        model = GPT(args.api_key, model_name, temperature, base_url)
        return model
    else:
        raise KeyError(f"Model {model_name} not implemented")


class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, api_key, model_name, temperature=1.0, base_url=None):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.badrequest_count = 0
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url
        self.client = OpenAI(api_key = api_key, base_url = base_url)
        
    # 在 forward 函数中调用
    def get_response(self, **kwargs):
        try:
            # res = openai.chat.completions.create(**kwargs)
            res = self.client.chat.completions.create(**kwargs)
            return res
        except openai.APIConnectionError as e:
            print('APIConnectionError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.APIConnectionError as err:
            print('APIConnectionError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.RateLimitError as e:
            print('RateLimitError')
            time.sleep(10)
            return self.get_response(**kwargs)
        except openai.APITimeoutError as e:
            print('APITimeoutError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.BadRequestError as e:
            print('BadRequestError')
            self.badrequest_count += 1
            print('badrequest_count', self.badrequest_count)
            return None


    def forward(self, head=None, prompt=None, use_cache=True, logger=None, forward_use_logger=True, use_json_format=False):
        messages = []
        info = {} 

        if logger == None:
            forward_use_logger = False   

        if head != None:
            messages.append(
                {"role": "system", "content": head}
            )
        messages.append(
            {"role": "user", "content": prompt}
        )

        key = json.dumps([self.model_name, messages])
        
        if forward_use_logger == True:
            logger.info(f"Messages: {str(messages)[:LOG_MAX_LENGTH]}")
        
        if use_cache:
            cached_value = get_from_cache(key, logger, forward_use_logger)
            if cached_value is not None:
                if forward_use_logger == True:
                    logger.info("Cache Hit")
                else:
                    print("Cache Hit")
                return cached_value, None
        
        if forward_use_logger:
            logger.info("Cache Miss")
        else:
            print("Cache Miss")

        if use_json_format:
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        else:
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

        if response is None:
            info['response'] = None
            info['message'] = None
            return None, info
        else:
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            info = dict(response.usage)  # completion_tokens, prompt_tokens, total_tokens
            info['response'] = messages[-1]["content"]
            info['message'] = messages

            if forward_use_logger:
                logger.info(f"Response: {str(info['response'])[:LOG_MAX_LENGTH]}")
            # if use_cache:
            save_to_cache(key, info['response'], logger, forward_use_logger)
            if forward_use_logger:
                logger.info("Cache Saved")
            else:
                print("Cache Miss")

            return self.post_process_fn(info['response']), info


if __name__ == "__main__":
    # args = parse_args()
    # pprint(args)
    # # get model
    # model = get_model(args)
    # model.set_post_process_fn(prompter.post_process_fn)

    api_key = "sk-T1Jnoi2vQSo4UzAw7c4e64Be4d774778B70416116e745dEd"
    base_url = "https://api.juheai.top/v1/"
    model_name = "gpt-4o"
    summarizer = GPT(api_key=api_key, model_name=model_name, base_url=base_url)


    # 设置 system-level prompt 和用户级 prompt 列表
    head = "You are a helpful assistant."
    prompt = "What is the capital of France?"

    # 调用 forward 方法获取响应
    response, info = summarizer.forward(head, prompt)

    # 打印响应和信息
    print("Response:", response)
    print("Info:", info)