# How to create a custom prompt template

假设我们希望LLM生成给定函数名称的英语解释。为了完成此任务，我们将创建一个自定义提示模板，**该模板将函数名作为输入，并格式化提示模板以提供函数的源代码。**

一共有两种不同的提示模版：

基本上有两种不同的提示模板可用——字符串提示模板和聊天提示模板。 字符串提示模板提供字符串格式的简单提示，而聊天提示模板生成更结构化的提示以与聊天 API 一起使用。 

下面的例子将使用字符串提示模板创建自定义提示。 要创建自定义字符串提示模板，有两个要求：它具有一个 input_variables 属性，该属性公开提示模板期望的输入变量。 它公开了一个格式方法，该方法接受与预期的 input_variables 相对应的关键字参数并返回格式化的提示。 我们将创建一个自定义提示模板，将函数名称作为输入并格式化提示以提供函数的源代码。

为此，让我们首先创建一个函数，该函数将返回给定函数名称的函数的源代码。

```python
# inspect 检查源代码和类布局
import inspect

def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)
```
接下来，我们将创建一个自定义提示模板，该模板接受函数名作为输入，并格式化提示模板以提供函数的源代码。

```python

from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """ A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function. """

	# 校验
    @validator("input_variables")
    def validate_input_variables(cls, v):
        """ Validate that the input variables are correct. """
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v
	
	# 格式生成
    def format(self, **kwargs) -> str:
        # Get the source code of the function
        source_code = get_source_code(kwargs["function_name"])

        # Generate the prompt to be sent to the language model
        prompt = f"""
        Given the function name and source code, generate an English language explanation of the function.
        Function Name: {kwargs["function_name"].__name__}
        Source Code:
        {source_code}
        Explanation:
        """
        return prompt
    
    def _prompt_type(self):
        return "function-explainer"
```

```python
fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])

# Generate a prompt for the function "get_source_code"
prompt = fn_explainer.format(function_name=get_source_code)
print(prompt)
```


# How to create a prompt template that uses **few shot examples**

下面的例子将使用`FewShotPromptTemplate`类来创建一个使用fewshot示例的提示模板。这个类要么接受一组示例，要么接受一个`ExampleSelector`对象。

## Using an example set

```python   
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
  {
    "question": "Who lived longer, Muhammad Ali or Alan Turing?",
    "answer": 
"""
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
"""
  },
  {
    "question": "When was the founder of craigslist born?",
    "answer": 
"""
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
"""
  },
  {
    "question": "Who was the maternal grandfather of George Washington?",
    "answer":
"""
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
"""
  },
  {
    "question": "Are both the directors of Jaws and Casino Royale from the same country?",
    "answer":
"""
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
"""
  }
]
```

## Create a formatter for the few shot examples
配置formatter，将少fewshot示例格式化为字符串。这个格式化程序应该是一个`PromptTemplate`对象。

```python
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

print(example_prompt.format(**examples[0]))
```

## Feed examples and formatter to FewShotPromptTemplate

将示例和格式化程序传递给`FewShotPromptTemplate`类，以创建一个使用fewshot示例的提示模板。

```python
prompt = FewShotPromptTemplate(
    examples=examples, 
    example_prompt=example_prompt, 
    suffix="Question: {input}", 
    input_variables=["input"]
)


print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

## Using an example selector

**Feed examples into `ExampleSelector`**

重用上一节中的示例集和格式化程序，只是接下来不是将示例直接提供给 `FewShotPromptTemplate` 对象，而是将它们提供给 `ExampleSelector` 对象。
在本教程中，我们将使用 `SemanticSimilarityExampleSelector` 类。 此类根据与输入的相似性选择少量镜头示例。 它使用一个嵌入模型来计算输入和少数镜头示例之间的相似性，以及一个向量存储来执行最近邻搜索。

这里Chroma安装经常出现错误 见常见错误及解决方法
```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1
)

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

**Feed example selector into `FewShotPromptTemplate`**

```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt, 
    suffix="Question: {input}", 
    input_variables=["input"]
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

## How to work with partial Prompt Templates    使用部分提示模版

Prompt Templates 是一个带有.format方法的类，该方法接受一个键值映射并返回一个字符串(prompt)以传递给语言模型。与其他方法一样，对提示模板进行“局部化”是有意义的——例如传入所需值的子集，以便创建一个只需要剩余值子集的新提示模板。

LangChain中有实现这个的两种方式:(1)带有字符串值，(2)带有返回字符串值的函数。

这两种方式支持不同的用例。

### Partial With Strings

想要局部化提示模板的一个常见用例是，您在其他变量之前获得了一些变量。例如，假设您有一个提示模板，它需要两个变量`foo`和`baz`。如果您在chain的早期获得`foo`值，但稍后获得`baz`值，那么等到两个变量位于同一位置时才将它们传递给提示模板可能会很烦人。可以使用`foo`值对提示模板进行局部化，然后传递局部化的提示模板，然后使用它。下面是这样做的一个例子:

```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template="{foo}{bar}", input_variables=["foo", "bar"])
partial_prompt = prompt.partial(foo="foo");
print(partial_prompt.format(bar="baz"))
```


也可以只使用部分变量初始化prompt。
```python
prompt = PromptTemplate(template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"})
print(prompt.format(bar="baz"))
```
### Partial With Functions

另一个常见的用法是对函数进行偏导。这个用例是当你有一个变量，你知道你总是想以一种常见的方式获取。最典型的例子就是日期或时间。假设您有一个提示符，您总是希望显示当前日期。您不能在提示符中硬编码它，并且将它与其他输入变量一起传递有点烦人。在这种情况下，使用一个总是返回当前日期的函数对提示符进行局部化是非常方便的。

```python
from datetime import datetime

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}", 
    input_variables=["adjective", "date"]
);
partial_prompt = prompt.partial(date=_get_datetime)
print(partial_prompt.format(adjective="funny"))
```

```python
prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}", 
    input_variables=["adjective"],
    partial_variables={"date": _get_datetime}
);
print(prompt.format(adjective="funny"))
```

## How to serialize prompts 序列化提示

这一节是关于langchain中prompt的加载

下面是从磁盘加载prompt的入口点，使得加载任何类型的prompt都很容易。

```python
# All prompts are loaded through the `load_prompt` function.
from langchain.prompts import load_prompt
```
##  加载 PromptTemplate
**Loading from YAML**
```shell
!cat simple_prompt.yaml
```

```python
prompt = load_prompt("./exampledata/simple_prompt.yaml")
print(prompt.format(adjective="funny", content="chickens"))
```

**Loading from JSON**

```python
prompt = load_prompt("simple_prompt.json")
print(prompt.format(adjective="funny", content="chickens"))
```

