# Prompt

本节包含与提示相关的所有内容。 提示是传递给语言模型的值。 此值可以是string（for LLM）或 list of messages（for chatmodel）。 这些提示的数据类型相当简单，但它们的构造比较复杂。

 LangChain提供： 
 
* 字符串提示和消息提示的标准接口 
 
* 字符串提示模板和消息提示模板的标准接口
 
* Example Selectors：将示例插入提示的方法，以便语言模型遵循 

* OutputParsers： 将指令作为语言模型输出信息的格式插入提示的方法，以及随后将该字符串输出解析为某种格式的方法。 

## Prompt Templates

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")
chat_prompt = ChatPromptTemplate.from_template("tell me a joke about {subject}")
string_prompt_value = string_prompt.format_prompt(subject="soccer")
chat_prompt_value = chat_prompt.format_prompt(subject="soccer")
```

传递给lLM时候调用的内容
```python
string_prompt_value.to_string()
# -->'tell me a joke about soccer'
chat_prompt_value.to_string()
# --> 'Human: tell me a joke about soccer'

```

传递给chatmodel时候调用的内容
```python
string_prompt_value.to_messages()
# --->[HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]
chat_prompt_value.to_messages()
# -->[HumanMessage(content='tell me a joke about soccer', additional_kwargs={}, example=False)]
```