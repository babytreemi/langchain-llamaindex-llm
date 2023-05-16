# How to cache LLM calls
缓存调用指南 

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

## Redis Cache
Use [Redis](https://redis.io/) to cache prompts and responses
```python
# We can do the same thing with a Redis cache
# (make sure your local Redis instance is running first before running this example)
from redis import Redis
from langchain.cache import RedisCache

langchain.llm_cache = RedisCache(redis_=Redis())

# We can do the same thing with a Redis cache
# (make sure your local Redis instance is running first before running this example)
from redis import Redis
from langchain.cache import RedisCache

langchain.llm_cache = RedisCache(redis_=Redis())
```

后面的测试方式和前面一样 `llm('input')`

## Semantic Cache
语义缓存 使Redis缓存prompts 和 responses， 并且基于语义相似度评估命中情况（ evaluate hits based on semantic similarity.）

我们可以使用GPTCache进行精确匹配缓存或基于语义相似度缓存结果


一个精确匹配的例子:

```python
# TODO：这段代码运行llm('input')会报错 官方文档错误 已经使用get_hashed_name解决，具体解决方法参考本项目文档 常见错误及解决方式
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache

# Avoid multiple caches using the same file, causing different llm model caches to affect each other

def init_gptcache(cache_obj: Cache, llm str):
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{llm}"),
    )

langchain.llm_cache = GPTCache(init_gptcache)
```

一个相似匹配的例子：

```python
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.cache import GPTCache

# Avoid multiple caches using the same file, causing different llm model caches to affect each other

def init_gptcache(cache_obj: Cache, llm str):
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{llm}")

langchain.llm_cache = GPTCache(init_gptcache)
```

## SQLAlchemy Cache
使用SQLAlchemyCache缓存SQLAlchemy支持的任何SQL数据库。
可以定义自己的声明式SQLAlchemyCache子类来定制用于缓存的模式。例如，使用Postgres支持高速全文提示索引

```python
# Column, Integer, String, Computed, Index, Sequence 是用于定义数据库表格和字段的类。
# create_engine 函数用于创建一个连接到数据库的引擎。
# declarative_base 函数，这个函数用于创建一个新的基类，该基类的子类可以自动与一个数据库表关联。
# 
from sqlalchemy import Column, Integer, String, Computed, Index, Sequence
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import TSVectorType
from langchain.cache import SQLAlchemyCache

# 创建了一个新的基类，名为 Base
Base = declarative_base()

# 创建一个从从base继承的类，这个类表示一个数据库表，该表用于存储 LLM 的生成结果。
class FulltextLLMCache(Base):  # type: ignore
	"""Postgres table for fulltext-indexed LLM Cache"""

	# 表的名称
	__tablename__ = "llm_cache_fulltext"
    # 定义名为‘id’的列
	id = Column(Integer, Sequence('cache_id'), primary_key=True)
    # 定义名为‘prompt’的列，不允许为空
	prompt = Column(String, nullable=False)
	# 定义名为‘llm’的列，不允许为空
	llm = Column(String, nullable=False)

	idx = Column(Integer)
	response = Column(String)

	# 定义名为‘prompt_tsv’的列，类型为TSVectorType，这个列是一个计算列，其值根据 llm 和 prompt 列的值计算得出。计算的方法是将 llm 和 prompt 列的值连接起来，然后使用 `to
	prompt_tsv = Column(TSVectorType(), Computed("to_tsvector('english', llm || ' ' || prompt)", persisted=True))
 	table_args__ = (
        Index("idx_fulltext_prompt_tsv", prompt_tsv, postgresql_using="gin"),
    )

engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine, FulltextLLMCache)

```

注意需要运行对应的**PostgreSQL**服务器

## Optional Caching

```python
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)
```

## Optional Caching in Chains

```python
llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)
```

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain

text_splitter = CharacterTextSplitter()
```

```python
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)
```

```python
from langchain.docstore.document import Document

# 从texts列表中取前三个元素，为每个元素创建一个Document对象，然后把这些对象放在一个列表中。
docs = [Document(page_content=t) for t in texts[:3]]
from langchain.chains.summarize import load_summarize_chain
```

```python
chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)

chain.run(docs)
```

当我们再次运行 `chain.run()时候，会发现运行得更快，但最终的答案是不同的。这是由于在 map 步骤缓存，而在 reduce 步骤不进行缓存。