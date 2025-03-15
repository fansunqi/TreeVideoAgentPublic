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

We obtain the dataset annotations and extracted captions from the File [LLoVi](https://drive.google.com/file/d/13M10CB5ePPVlycn754_ff3CwnpPtDfJA/view?usp=drive_link) provide. We have already placed them in ```data/egoschema```.

```
python main.py 
```



2. Xxx

## TODO:

+ example_summary 和 qa_exmple_summary 什么时候指定

+ 解决下面的报错：

  ```
  Traceback (most recent call last):
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/video_seg.py", line 49, in extract_videoseg_from_descriptions
      start, end = map(int, duration.split('-'))  # 解析 'start-end' 格式
      ^^^^^^^^^^
  ValueError: invalid literal for int() with base 10: 'limit'
  
  During handling of the above exception, another exception occurred:
  
  Traceback (most recent call last):
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 677, in <module>
      main(args)
      ~~~~^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 648, in main
      for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
               ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/tva_env/lib/python3.13/site-packages/tqdm/std.py", line 1181, in __iter__
      for obj in iterable:
                 ^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 619, in result_iterator
      yield _result_or_cancel(fs.pop())
            ~~~~~~~~~~~~~~~~~^^^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 317, in _result_or_cancel
      return fut.result(timeout)
             ~~~~~~~~~~^^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 456, in result
      return self.__get_result()
             ~~~~~~~~~~~~~~~~~^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 401, in __get_result
      raise self._exception
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/thread.py", line 59, in run
      result = self.fn(*self.args, **self.kwargs)
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 648, in <lambda>
      for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
                                           ~~~~~~~~~~~~~~~~^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 537, in run_one_question
      select_process(formatted_question, sample_idx, sampled_caps, num_frames,
      ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     step, args, all_sample_idx, caps, video_segments, select_fn)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 445, in select_process
      selected_video_segments = extract_videoseg_from_descriptions(selected_descriptions)
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/video_seg.py", line 53, in extract_videoseg_from_descriptions
      start = int(duration)
  ValueError: invalid literal for int() with base 10: '155-limit'
  ```

  ```
  Traceback (most recent call last):
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 677, in <module>
      main(args)
      ~~~~^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 648, in main
      for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
               ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/tva_env/lib/python3.13/site-packages/tqdm/std.py", line 1181, in __iter__
      for obj in iterable:
                 ^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 619, in result_iterator
      yield _result_or_cancel(fs.pop())
            ~~~~~~~~~~~~~~~~~^^^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 317, in _result_or_cancel
      return fut.result(timeout)
             ~~~~~~~~~~^^^^^^^^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 456, in result
      return self.__get_result()
             ~~~~~~~~~~~~~~~~~^^
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/_base.py", line 401, in __get_result
      raise self._exception
    File "/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/concurrent/futures/thread.py", line 59, in run
      result = self.fn(*self.args, **self.kwargs)
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 648, in <lambda>
      for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
                                           ~~~~~~~~~~~~~~~~^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 537, in run_one_question
      select_process(formatted_question, sample_idx, sampled_caps, num_frames,
      ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     step, args, all_sample_idx, caps, video_segments, select_fn)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/main.py", line 445, in select_process
      selected_video_segments = extract_videoseg_from_descriptions(selected_descriptions)
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/video_seg.py", line 46, in extract_videoseg_from_descriptions
      duration = get_duration(description)
    File "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgentPublic/util.py", line 160, in get_duration
      for key in description.keys():
                 ^^^^^^^^^^^^^^^^
  AttributeError: 'str' object has no attribute 'keys'
  ```

  
