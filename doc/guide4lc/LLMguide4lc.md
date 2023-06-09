# 语言模型
语言模型有两种不同的子类型：

1. LLMs：接收文本并返回文本

2. ChatModels：接收聊天消息并返回聊天消息
   
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

### How to use the async API for LLMs  异步调用 
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

### How to write a custom LLM wrapper 编写LLM包装器
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

### How (and why) to use the the human input LLM
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

### How to cache LLM calls 缓存LLM调用

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
```python
llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)

chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)

```
### How to serialize LLM classes
这部分是关于如何在磁盘上写入和读取 LLM 配置（例如，提供商、温度等）。
```python
from langchain.llms import OpenAI
from langchain.llms.loading import load_llm
```
从磁盘加载llm：  LLM 可以以两种格式保存在磁盘上：json 或 yaml。无论扩展名如何，它们都以相同的方式加载。
```shell
cat llm.json
```
结果：
```
{
    "model_name": "text-davinci-003",
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "n": 1,
    "best_of": 1,
    "request_timeout": null,
    "_type": "openai"
}

```

```python
llm = load_llm("llm.json")
```
或者
```shell
cat llm.yaml
```
结果
```
_type: openai
best_of: 1
frequency_penalty: 0.0
max_tokens: 256
model_name: text-davinci-003
n: 1
presence_penalty: 0.0
request_timeout: null
temperature: 0.7
top_p: 1.0

```

```python
llm = load_llm("llm.yaml")
```

如果您想从内存中的LLM转换为serialized版本，可以通过调用.save方法轻松实现，同时支持json和yaml。
```python
llm.save("llm.json")
# llm.save("llm.yaml")
```

### How to stream LLM and Chat Model responses 流式响应

实现LLM和Chat Model的流式响应

为什么要使用流式响应：主要关注用户体验 在聊天应用程序的上下文中，当LLM生成令牌时，它可以立即提供给用户。虽然这不会改变从问题提交到完全响应的端到端执行时间，但它通过向用户显示LLM正在取得进展，大大减少了感知到的延迟。

LangChain为llm提供流支持。目前，支持`OpenAI`、`ChatOpenAI`和`ChatAnthropic`实现的流，但对其他LLM实现的流支持正在路线图中。使用可以实现`on_llm_new_token`的`CallbackHandler`。下面的例子使用`StreamingStdOutCallbackHandler`：

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = llm("Write me a song about sparkling water.")
```

如果使用`generate`，我们仍然可以访问最后的`LLMResult`。但是，`token_usage`目前不支持流。
```python
llm.generate(["Tell me a joke."])
```
下面是一个使用ChatOpenAI聊天模型实现的例子:
```python
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
```
下面是 `ChatAnthropic` 聊天模型实现的示例，使用`claude` 模型:
```python
chat = ChatAnthropic(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
```
### How to track token usage
目前只适配`openai`API

```python
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# 跟踪一个调用
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)

# 按顺序跟踪多个调用
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    result2 = llm("Tell me a joke")
    print(cb.total_tokens)

```

在 `chain` 和 `agent` 中跟踪所有会话

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with get_openai_callback() as cb:
    response = agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

```
### Integrations 关于不同LLM提供商的集成指南
The examples here are all “how-to” guides for how to integrate with various LLM providers.
这里仅展示常用的，详细请见：[Integrations](https://python.langchain.com/en/latest/modules/models/llms/integrations.html)

1. **AI21** <https://studio.ai21.com/account/account>

    `pip install ai21`

2. **Aleph Alpha**  主要针对[Luminous Model Family](https://docs.aleph-alpha.com/docs/introduction/luminous/)
   
    <https://docs.aleph-alpha.com/docs/account/#create-a-new-token>
   
   `pip install aleph-alpha-client`

  
3. **Anyscale** <https://docs.anyscale.com/productionize/services-v2/get-started>

4. **Azure OpenAI** <https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/>
    
5. **Banana**  <https://app.banana.dev/>
   
    `pip install banana-dev`

6. **CerebriumAI**  <https://dashboard.cerebrium.ai/login>
   
   `Cerebrium ` 是 AWS Sagemaker 的一个替代方案

   `pip3 install cerebrium`

7. **Cohere**  <https://dashboard.cohere.ai/>
   
    `pip install cohere`

8. **DeepInfra**  <https://deepinfra.com/dash/deployments>

9. **ForefrontAI** <https://docs.forefront.ai/forefront/api-reference/authentication>

   `Forefront` 微调和使用开源大型语言模型。 

10. **GooseAI** <https://goose.ai/>
    
     `GooseAI` is a fully managed NLP-as-a-Service, delivered via API. 
11. **GPT4All** 项目地址：<https://github.com/nomic-ai/gpt4all>
 
    GPT4All是一个开源聊天机器人生态系统，在大量干净的助手数据（包括代码、故事和对话）上进行训练。

    支持本地运行
    ```python
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    # If you want to use GPT4ALL_J model add the backend parameter
    llm = GPT4All(model=local_path, backend='gptj', callbacks=callbacks, verbose=True)
    ```

12. **Hugging Face Hub** <https://huggingface.co/docs/api-inference/quicktour#get-your-api-token>
    
    `pip install huggingface_hub > /dev/null`

    tips: `> /dev/null`：这是一个shell命令，意思是将命令的输出（也就是`pip install huggingface_hu`b的输出）重定向到`/dev/null`。`/dev/null`是一个特殊的文件，所有写入它的数据都会被丢弃，也就是说，这个命令的输出不会显示在命令行界面上。
    ```python
    repo_id = "google/flan-t5-xl" 
    # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
    ```

13. **Hugging Face Local Pipelines**
    
    `HuggingFacePipeline` class:在本地运行huggingface模型

    `pip install transformers > /dev/null

    ```python
    from langchain import HuggingFacePipeline
    
    llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":64})
    ```
14. **Huggingface TextGen Inference**

    Text Generation Inference是一个用于文本生成推理的Rust, Python和gRPC服务器。HuggingFace在生产中使用，为llm api-inference widges 提供动力。

    `pip3 install text_generation  `
        
    ```python

    llm = HuggingFaceTextGenInference(
        inference_server_url='http://localhost:8010/',\
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        )
    llm("What did foo say about bar?")
    ```
15. **Structured Decoding with JSONFormer**
    
    JSONFormer是一个库，它封装了本地HuggingFace pipeline模型，用于对JSON模式的一个子集进行结构化解码。
    它通过填充结构令牌，然后从模型中采样内容令牌来工作。

    `pip install --upgrade jsonformer > /dev/null`

    - [ ] 优化langchain-chatglm的result

    ```python
    json_former = JsonFormer(json_schema=decoder_schema, pipeline=hf_model)
    results = json_former.predict(prompt, stop=["Observation:", "Human:"])
    print(results)
    ```
16. **Llama-cpp** <https://github.com/abetlen/llama-cpp-python> 
    
    不需要`API_token`

    `pip install llama-cpp-python`

17. **Manifest** 

    有关`manifest`的更多详细信息，以及如何将其与本例中的本地hugginface模型一起使用，参阅<https://github.com/HazyResearch/manifest>

    example: <https://github.com/HazyResearch/manifest/blob/main/examples/langchain_chatgpt.ipynb>

    `pip install manifest-ml`
    

    可以用于比较 HF Models

18. **Modal**  <https://modal.com/docs/guide>
    
    Modal Python Library提供了从本地计算机上的Python脚本方便地按需访问无服务器云计算的功能。Modal本身不提供任何llm，只提供基础设施。

    `pip install modal-client`

    `modal token new`

19. **NLP Cloud** <https://nlpcloud.com/>

    get a token: <https://docs.nlpcloud.com/#authentication>

    NLP Cloud为NER、情感分析、分类、摘要、释义、语法和拼写纠正、关键字和关键短语提取、聊天机器人、产品描述和广告生成、意图分类、文本生成、图像生成、博客帖子生成、代码生成、问答、自动语音识别、机器翻译、语言检测、语义搜索、语义相似度、标记化、POS标记、嵌入和依赖项解析。它已经为生产做好了准备，通过REST API提供服务

    `pip install nlpcloud`

20. **OpenAI** <https://platform.openai.com/account/api-keys>
21. **Petals** <https://github.com/bigscience-workshop/petals>
    
    用的是huggingface的apikey <https://huggingface.co/docs/api-inference/quicktour#get-your-api-token>

22. **PipelineAI** <https://docs.pipeline.ai/docs/cloud-quickstart>
    
    使用PipelineAI在云端大规模运行机器学习模型。还提供了对几个LLM模型的API访问:<https://www.mystic.ai/pipeline-catalyst>。

    `pip install pipeline-ai`

23. **PredictionGuard**
    这是一个wrapper 
    
    使用参考：<https://python.langchain.com/en/latest/modules/models/llms/integrations/predictionguard.html>

24. **PromptLayer OpenAI**  <https://promptlayer.com/>
    
    `PromptLayer`是第一个允许您跟踪、管理和共享GPT提示工程的平台。`PromptLayer`充当代码和`OpenAI` python库之间的中间件。`PromptLayer`记录所有`OpenAI API`请求，允许在`PromptLayer`仪表板中搜索和探索请求历史以及跟进模型性能。

    [example1](https://python.langchain.com/en/latest/modules/models/llms/integrations/promptlayer_openai.html)

    [example2](https://python.langchain.com/en/latest/ecosystem/promptlayer.html)

25. Structured Decoding with RELLM <https://github.com/r2d4/rellm>
    
    RELLM是一个库，它封装了用于结构化解码的本地HuggingFace管道模型。
    它的工作原理是一次生成一个令牌。在每一步中，它都会屏蔽不符合所提供的部分正则表达式的令牌
    
    `pip install rellm > /dev/null`
26. **Replicate**
    
    replication在云端运行机器学习模型。langchain有一个开源模型库，你可以用几行代码运行它。如果您正在构建自己的机器学习模型，则可以轻松地大规模部署它们。(也可以用cv或者多模态模型)

    模型库: <https://replicate.com/explore>

    具体例子参考：<https://python.langchain.com/en/latest/modules/models/llms/integrations/replicate.html> (这是一个关于图像生成的例子)

27. **Runhouse** <https://github.com/run-house/runhouse>
    
    Runhouse允许跨环境和用户的远程计算和数据。参见[Runhouse文档](https://runhouse-docs.readthedocs-hosted.com/en/latest/)。

    [例子](https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html)介绍了如何使用LangChain和Runhouse与托管在您自己的GPU上的模型进行交互，或者在AWS、GCP、AWS或Lambda上的GPU。

    Note: Code uses `SelfHosted` name instead of the `Runhouse`.

    `pip install runhouse`


28. **SageMakerEndpoint**  <https://aws.amazon.com/cn/sagemaker/>
    
    Amazon SageMaker是一个可以为任何用例构建、训练和部署机器学习(ML)模型的系统，具有完全托管的基础设施、工具和工作流。

29. **StochasticAI** <https://docs.stochastic.ai/docs/introduction/>

    get the API_KEY and the API_URL: <https://app.stochastic.ai/login>
    
    简化深度学习模型的生命周期。从上传和版本化模型，到训练、压缩和加速，再到将其投入生产。

30. **Writer** <https://writer.com/>
    
    Writer是一个生成不同语言内容的平台。

    获得WRITER_API_KEY <https://dev.writer.com/docs>

