# How to cache LLM calls
缓存调用结果指南

## In Memory Cache
```python
from langchain.llms import OpenAI

import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
```

```python 
%%time
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
```

```python
%%time
# The second time it is, so it goes faster
llm("Tell me a joke")
```

## SQLite Cache
```python
# 在jupyter中删除文件
!rm .langchain.db
```

```python
# 在其他解释器中删除文件
import os

# Check if the file exists before trying to delete it
if os.path.exists(".langchain.db"):
    os.remove(".langchain.db")
else:
    print("The file does not exist")


#或者
# from pathlib import Path

# db_file = Path(".langchain.db")
# if db_file.exists():
#     db_file.unlink()
# else:
#     print("The file does not exist")、

```
SQLite cache 示例代码
```python
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
````

```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm("Tell me a joke")
```

```Python
%%time
# The second time it is, so it goes faster
llm("Tell me a joke")
```
