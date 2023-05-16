# Getting Started
本文档只展示一些核心代码，具体代码见：<https://github.com/babytreemi/langchain-llamaindex-llm/tree/main/doc/use_detail4lc>

快速体验langchain的[notebook](https://github.com/babytreemi/langchain-llamaindex-llm/tree/main/notebooks#:~:text=QuickstartGuide.-,ipynb,-QuickstartGuide11.ipynb)

langchain为模型提供了一个标准统一的接口

有两种主要类型的模型：

**1. 语言模型**：适合文本生成

**2. text-embedding模型**：适用于将文本转换为数字表示

## 语言模型
语言模型有两种不同的子类型：

1. LLMs：文本并返回文本

2. ChatModels：收聊天消息并返回聊天消息
   
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

# text -> text interface
llm.predict("say hi!")
chat_model.predict("say hi!")

# messages -> message interface
from langchain.schema import HumanMessage
llm.predict_messages([HumanMessage(content="say hi!")])
chat_model.predict_messages([HumanMessage(content="say hi!")])

```
## LLM 
### 重要函数
介绍了如何在LangChain中使用LLM类。
LLM类是为与LLM接口而设计的类。有很多LLM提供商(OpenAI, Cohere, hug Face等)-这个类的目的是为他们提供一个标准的接口。在文档的这一部分中，我们将重点关注一般的LLM功能。有关使用特定LLM包装器的详细信息，请参阅 [How-To ](https://python.langchain.com/en/latest/modules/models/llms/how_to_guides.html)中的示例。

最基本的功能：输入一个字符串 输出一个字符串

```python
```python
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
llm("Tell me a joke")
# -->'\n\nWhy did the chicken cross the road?\n\nTo get to the other side.'
```
拓展功能：输入一个字符串列表 

```python
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)
len(llm_result.generations)
# --> 30
llm_result.generations[0]
llm_result.generations[-1]
```

访问相关信息
```python
llm_result.llm_output
# -->{'token_usage': {'completion_tokens': 3903,'total_tokens': 4023 'prompt_tokens': 120}}

```
利用`tiktoken`查看tonken数量:
```python
llm.get_num_tokens("what a joke")

# --> 3
```
### Generic Functionality 通用功能

#### How to use the async API for LLMs  异步调用 
**什么是异步：** 按照这种设计编写的代码使得程序能够要求一个任务与先前的一个（或多个）任务一起执行，而无需为了等待它们完成而停止执行。当后来的任务完成时，程序将使用约定好的机制通知先前的任务，以便让它知道任务已经完成，以及如果有结果存在的话，这个结果是可用的。

LangChain 通过利用 `asyncio` 库为 LLM 提供异步支持。 异步支持对于同时调用多个 LLM 特别有用，因为这些调用是网络绑定的。 目前，支持 `OpenAI`、`PromptLayerOpenAI`、`ChatOpenAI` 和 `Anthropic`，但对其他 LLM 的异步支持也在路线图上。 您可以使用 `generate` 方法异步调用 OpenAI LLM。

```python
# 演示 Python 的并发和串行执行的时间差异。
# time（用于计时），asyncio（用于并发编程）和 async（用于异步编程）是内置的。
import time
import asyncio

from langchain.llms import OpenAI

# generate_serially() 函数创建了一个 OpenAI 实例，并循环10次，每次都调用其 generate 方法来生成文本，然后将生成的文本打印出来。这个函数是串行运行的，意味着每次只有在前一个生成任务完成后，才会开始下一个任务。
def generate_serially():
    llm = OpenAI(temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        print(resp.generations[0][0].text)


# async_generate(llm) 是一个协程函数，它接收一个 OpenAI 实例作为参数，然后并发地调用其 agenerate 方法来生成文本，并将生成的文本打印出来。
async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)

# generate_concurrently() 函数创建了一个 OpenAI 实例，并创建了10个 async_generate 任务，然后使用 asyncio.gather() 函数并发地运行这些任务。这个函数是并发运行的，意味着所有的生成任务都是同时开始的。
async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)

# 并行
s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
await generate_concurrently() 
elapsed = time.perf_counter() - s
print('\033[1m' + f"Concurrent executed in {elapsed:0.2f} seconds." + '\033[0m')

# 串行
s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print('\033[1m' + f"Serial executed in {elapsed:0.2f} seconds." + '\033[0m')
```

#### How to write a custom LLM wrapper 编写LLM包装器
编写一个包装器，以便使用自己的LLM模型和langchain支持的其他LLM
定制LLM包装只要实现一件事：

`_call`方法，它接受一个字符串和一些可选的停止词，并返回一个字符串

可选：

`_identifying_params`属性，用于帮助打印该类。应该返回一个字典。

下面是一个简单的示例

```python
# typing 模块支持类型提示
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class CustomLLM(LLM):
    
    n: int

	# 包装器 将方法转换为属性
    @property
		#-> str: 这是一个类型注解，表示这个方法返回的值是一个字符串类型。
		# 尽管 _llm_type() 方法可以被调用，但由于它的名称以单下划线 _ 开头，所以它应该被视为类的非公开方法。这意味着，除非你确切知道你在做什么，否则你不应该在类的外部调用这个方法。
       
    def _llm_type(self) -> str:
		 return "custom"
    
	# 定义一个_call方法，该方法是在 CustomLLM 类中定义的。这个方法的作用是根据输入的 prompt 返回其前 n 个字符。
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]
    
    @property
	# 返回一个字典
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}


llm = CustomLLM(n=10)
llm("This is a foobar thing")
	# --> 'This is a '	
print(llm)

	# CustomLLM
	# Params: {'n': 10}
```

### How (and why) to use the fake LLM 用fakeLLM进行测试
这允许您模拟对 LLM 的调用并模拟如果 LLM 以某种方式响应会发生什么。

```python
from langchain.llms.fake import FakeListLLM

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["python_repl"])

responses=[
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4"
]
llm = FakeListLLM(responses=responses)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("whats 2 + 2")
```

#### How (and why) to use the the human input LLM
与假LLM类似，LangChain提供了一个伪LLM类，可用于测试、调试或教育目的。这允许您模拟对LLM的调用，并模拟如果人类收到提示将如何响应。

Using `HumanInputLLM` in an agent.

- [ ] 代码运行不通 官方文档有误 

```python
from langchain.llms.human import HumanInputLLM

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["wikipedia"])
llm = HumanInputLLM(prompt_func=lambda prompt: print(f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"))

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What is 'Bocchi the Rock!'?")
```

#### How to cache LLM calls 缓存LLM调用

官方文档：[Caching LLM Calls](https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html#how-to-cache-llm-calls)

更多细节：

- [ ] 整合 [GPTCache](https://github.com/zilliztech/GPTCache)


**In Memory Cache**

```python
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
```
**SQLite Cache**

```python
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
```

**Redis Cache**

Standard Cache

```python
from redis import Redis
from langchain.cache import RedisCache

langchain.llm_cache = RedisCache(redis_=Redis())
```
Semantic Cache

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.cache import RedisSemanticCache


langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings()
)
```

**GPTCache**

官方doc代码有错误，修正见常见错误及解决方案

 exact match

 ```python
 from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache

# Avoid multiple caches using the same file, causing different llm model caches to affect each other

def init_gptcache(cache_obj: Cache, llm: str):
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{llm}"),
    )

langchain.llm_cache = GPTCache(init_gptcache)

``` 
similarity caching

```python

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.cache import GPTCache

# Avoid multiple caches using the same file, causing different llm model caches to affect each other

def init_gptcache(cache_obj: Cache, llm: str):
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{llm}")

langchain.llm_cache = GPTCache(init_gptcache)
```

**SQLAlchemy Cache**

```python
engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine, FulltextLLMCache)
```

**Optional Caching**

针对特定的LLM，可以选择性地启用缓存。这可以通过在LLM的初始化中设置cache_enabled=True来完成。

```python	
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)
```
、
**Optional Caching in Chains**

关闭chain中特定节点的缓存