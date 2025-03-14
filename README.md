# 🌲 TreeVideoAgent

## News and Todo 🗓️

- [ ] Release Code for Demo

- [ ] Release Code for EgoSchema

- [ ] Release Code for NExT-QA

## Installation Steps 🛠️

Our TreeVideoAgent does not require many computational resources; it can run on a personal computer without GPU.

1. Clone the repository 📦:

   ```python
   git clone git@github.com:fansunqi/TreeVideoAgentPublic.git
   cd TreeVideoAgentPublic
   ```

2. Create a virtual environment 🧹 and install the dependencies 🧑‍🍳:

   ```python
   python3 -m venv tva_env
   source tva_env/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your API key 🗝️:

   Obtain an OpenAI API key and set your ```OPENAI_API_KEY``` and ```OPENAI_BASE_URL``` as environmental variables in  ```~/.zshrc``` or ```~/.bashrc```. In the ```main.py```, we will use the following codes to obtain the API key and base URL:

   ```
   api_key = os.getenv("OPENAI_API_KEY")
   base_url = os.getenv("OPENAI_BASE_URL")
   ```

## QuickStart 🚀

dhued

```
python main.py --dataset demo --output_base_path results/demo/ --logger_path results/demo/
```





- 05ad5736-88f5-42bb-ac9f-689e199c50de



## EgoSchema Experiments



## TODO:

+ example_summary 和 qa_exmple_summary 什么时候指定
