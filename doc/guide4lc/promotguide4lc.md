# Prompt

本节包含与提示相关的所有内容。 提示是传递给语言模型的值。 此值可以是string（for LLM）或 list of messages（for chatmodel）。 这些提示的数据类型相当简单，但它们的构造比较复杂。

关于本节的更多细节见：

 LangChain提供： 
 
* 字符串提示和消息提示的标准接口 
 
* 字符串提示模板和消息提示模板的标准接口
 
* Example Selectors：将示例插入提示的方法，以便语言模型遵循 

* OutputParsers： 将指令作为语言模型输出信息的格式插入提示的方法，以及随后将该字符串输出解析为某种格式的方法。 

## Getting Started

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

## Prompt Templates

提示模板指的是生成提示的可重复方式。它包含一个文本字符串(“模板”)，它可以从最终用户接收一组参数并生成prompt。
提示模板可能包含: 对语言模型的指令(instructions) + fewshot examples + question

```python
from langchain import PromptTemplate


template = """
I want you to act as a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# format格式化字符串模版
prompt.format(product="colorful socks")
# --> '\nI want you to act as a naming consultant for new companies.\nWhat is a good name for a company that makes colorful socks?\n'
```

我们可以使用 `PromptTemplate` 类创建简单的硬编码提示。提示模板可以采用**任意数量的输入变量**，并且可以格式化以生成提示。
```python
from langchain import PromptTemplate

# An example prompt with no input variables
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")

no_input_prompt
# -> PromptTemplate(input_variables=[], output_parser=None, partial_variables={}, template='Tell me a joke.', template_format='f-string', validate_template=True)
# format 格式化字符串模版
no_input_prompt.format()
# -> "Tell me a joke."


# An example prompt with one input variable
one_input_prompt = PromptTemplate(input_variables=["adjective"], template="Tell me a {adjective} joke.")
# 利用format传入funny
one_input_prompt.format(adjective="funny")
# -> "Tell me a funny joke."

# An example prompt with multiple input variables
multiple_input_prompt = PromptTemplate(
    input_variables=["adjective", "content"], 
    template="Tell me a {adjective} joke about {content}."
)
multiple_input_prompt.format(adjective="funny", content="chickens")
# -> "Tell me a funny joke about chickens."
```

如果您不希望手动指定`input_variables`，您也可以使用from_template类方法创建`PromptTemplate`。Langchain将根据传递的模板自动推断`input_variables`。

```python
template = "Tell me a {adjective} joke about {content}."

# 看这里看这里 简洁好多
prompt_template = PromptTemplate.from_template(template)
prompt_template.input_variables
# -> ['adjective', 'content']
prompt_template.format(adjective="funny", content="chickens")
# -> Tell me a funny joke about chickens.
```

默认情况下，`PromptTemplate` 会将提供的模板视为 Python f 字符串。您可以通过 `template_format` 参数指定其他**模板格式**：

```python

# 使用jinjia2模版
# Make sure jinja2 is installed before running this Copy code : pip install jinja2

jinja2_template = "Tell me a {{ adjective }} joke about {{ content }}"
# 增加template_format参数
prompt_template = PromptTemplate.from_template(template=jinja2_template, template_format="jinja2")

prompt_template.format(adjective="funny", content="chickens")
# -> Tell me a funny joke about chickens.
```

**Validate template**

默认情况下，`PromptTemplate`将通过检查`input_variables`是否与模板中定义的变量匹配来验证模板字符串。您可以通过将`validate_template`设置为`False`来禁用.

```python
template = "I am learning langchain because {reason}."

prompt_template = PromptTemplate(template=template, 
                                 input_variables=["reason", "foo"]) # ValueError due to extra variables
prompt_template = PromptTemplate(template=template, 
                                 input_variables=["reason", "foo"], 
                                 validate_template=False) # No error
```

**Serialize prompt template 序列化提示模版**
您可以将 `PromptTemplate` 保存到本地文件系统中的文件中。 langchain会通过文件扩展名自动推断文件格式。目前，langchain 支持将模板保存为 YAML 和 JSON 文件。

```python
 # Save to JSON file
prompt_template.save("awesome_prompt.json")

# load from JSON file
from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")

assert prompt_template == loaded_prompt
```

也可以从langchain hub加载模版

```python
from langchain.prompts import load_prompt

prompt = load_prompt("lc://prompts/conversation/prompt.json")
prompt.format(history="", input="What is 1 + 1?")
# --> No `_type` key found, defaulting to `prompt`.
# 'The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n\nHuman: What is 1 + 1?\nAI:'
```

**Pass few shot examples to a prompt template**

fewshot生成反义词的例子：

```python
from langchain import PromptTemplate, FewShotPromptTemplate

# 创造例子列表
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# 格式化例子
example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# 创造 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    # 前缀 通常是instructions
    prefix="Give the antonym of every input",
    # 后缀
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n",
)
print(few_shot_prompt.format(input="big"))
```

**Select examples for a prompt template 例子自动选择**


如果您有大量的示例，您可以使用`ExampleSelector`来选择对语言模型最有帮助的示例子集。这将帮助您生成一个更有可能产生良好响应的提示。
下面，我们将使用`LengthBasedExampleSelector`，它根据输入的长度选择示例。当您担心构造一个将超过上下文窗口长度的提示时，这很有用。对于较长的输入，它会选择更少的例子，而对于较短的输入，它会选择更多的例子。

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector

# 可供选择的例子
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

# example_selector
example_selector = LengthBasedExampleSelector(
    # 可供选择的例子
    examples=examples, 
    # This is the PromptTemplate being used to format the examples.
    example_prompt=example_prompt, 
    #这是格式化示例的最大长度。
    #长度由下面的get_text_length函数决定。
    max_length=25,
)

#使用' example_selector '来创建一个' FewShotPromptTemplate '。
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n",
)

print(dynamic_prompt.format(input="big"))
```
如果我们提供很长的输入，`LengthBasedExampleSelector` 将选择更少的示例。
