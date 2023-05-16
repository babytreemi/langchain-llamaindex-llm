# Chat Models

## 整体结构

导入所需要的包
```python
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
```
chatmodels的默认模型是`gpt-3.5-turbo`
```python
chat = ChatOpenAI(temperature=0)
```

您可以通过向聊天模型传递一条或多条消息来完成聊天。response是一条消息。

LangChain目前支持的消息类型有AIMessage、HumanMessage、SystemMessage和ChatMessage——ChatMessage接受任意角色参数。

一般只需要处理HumanMessage, AIMessage和SystemMessage

```python
# 1
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

# 2
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)
# --> AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False)
```
可以更进一步，使用 `generate` 为多组消息生成补全。这将返回一个带有附加消息参数的 `LLMResult`。

```python
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result

# --> [LLMResult(generations=[[ChatGeneration(text="J'adore la programmation.", generation_info=None, message=AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False))], [ChatGeneration(text="J'adore l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'adore l'intelligence artificielle.", additional_kwargs={}, example=False))]], llm_output={'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20, 'total_tokens': 77}, 'model_name': 'gpt-3.5-turbo'})
result.llm_output

# --> {'token_usage': {'prompt_tokens': 57, 'completion_tokens': 20,'total_tokens': 77},'model_name': 'gpt-3.5-turbo'}
```
## PromptTemplates

通过`MessagePromptTemplate`来使用模板，可以从一个或多个`MessagePromptTemplates`构建`ChatPromptTemplate`。您可以使用`ChatPromptTemplate的format_prompt`—返回一个`PromptValue`，可选择将其转换为字符串或Message对象，具体取决于您是否希望使用格式化的值作为llm或聊天模型的输入。为方便起见，在模板上公开了一个`from_template`方法。

```python
template="You are a helpful assistant that translates {input_language} to {output_language}."
# system_message_promp
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# human_message_prompt
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 拼接
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 从格式化的messages完成chat过程
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# --> AIMessage(content="J'adore la programmation.", additional_kwargs={}, example=False)
```

如果你想更直接地构造MessagePromptTemplate，你可以在外部创建一个PromptTemplate，然后传入，例如:

```python
prompt=PromptTemplate(
    template="You are a helpful assistant that translates {input_language} to {output_language}.",
    input_variables=["input_language", "output_language"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
```
## LLMChain

```python
chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")
```
## Streaming
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
```

## How to use few shot examples
### Alternating Human/AI messages 交替使用ai和human的消息
```python
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
chat = ChatOpenAI(temperature=0)
template="You are a helpful assistant that translates english to pirate."

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# examples
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")


human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
# get a chat completion from the formatted messages
chain.run("I love programming.")

# --> "I be lovin' programmin', me hearty!"
```

OpenAI提供了一个可选的`name参数`，他们还建议将该参数与系统消息结合使用，以执行fewshot。

```python
template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

#examples
example_human = SystemMessagePromptTemplate.from_template("Hi", additional_kwargs={"name": "example_user"})
example_ai = SystemMessagePromptTemplate.from_template("Argh me mateys", additional_kwargs={"name": "example_assistant"})

human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
# get a chat completion from the formatted messages
chain.run("I love programming.")

# --> "I be lovin' programmin', me hearty!"
```

## How to stream responses 流式输出

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])

```

## [Integrations](https://python.langchain.com/en/latest/modules/models/chat/integrations.html)