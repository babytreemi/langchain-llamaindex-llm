# langchain-llamaindex-llm
因为这两个库迭代太快➕看不懂大佬们写的代码➕英语不好➕踩坑太多➕环境配置总是冲突，所以自己造一个简单的仓库方便自己造福和我一样的小可怜

大模型预训练和服务调研：<https://github.com/babytreemi/all-about-llm>
github地址：<https://github.com/babytreemi/langchain-llamaindex-llm>

[gitbook](https://babytreemis-organization.gitbook.io/langchain_llama-index_llm/) 

目前本仓库主要是针对python，js后期可能会加入
## 介绍
**LangChain** 是一个用于开发由语言模型驱动的应用程序的框架。我们相信，强大和有区分度的应用程序不仅能通过 API 调用语言模型，而且应该可以：
1. Be data-aware: connect a language model to other sources of data
2. Be agentic: allow a language model to interact with its environment
## 概念
There are two main value props the LangChain framework provides:
**组件和链**
1. **Components**: LangChain provides modular abstractions for the components neccessary to work with language models. LangChain also has collections of implementations for all these abstractions. The components are designed to be easy to use, regardless of whether you are using the rest of the LangChain framework or not.
2. **Use-Case Specific Chains**: Chains can be thought of as assembling these components in particular ways in order to best accomplish a particular use case. These are intended to be a higher level interface through which people can easily get started with a specific use case. These chains are also designed to be customizable.



## 常用链接
langchain集成的列表：<https://langchain.com/integrations.html>

langchain Pythonrepo：<https://github.com/hwchase17/langchain>

langchain doc4python: <https://python.langchain.com/en/latest/>

langchain doc4js: <https://js.langchain.com/docs/>

使用google搜索需要设置SERPAPI_API_KEY，在这里注册：<https://serpapi.com/users/welcome>

Anthropic申请和注册 <https://www.anthropic.com/product>

langchain hub <https://github.com/hwchase17/langchain-hub>
## 常用命令和代码

### 1. install
`pip install langchain` 
or 
`conda install langchain -c conda-forge`

这里有个坑（包括安装其他的一些库时），多次遇到在conda环境中使用pip命令会找不到，

解决办法：
1. `conda install pip`，然后再用`pip`安装langchain
2. `python -m pip install langchain`

### 2.Environment Setup
最常用到openai api  安装对应的SDK
```shell
pip install openai
```
### 参数设定

`verbose=True`:prompt可见

配置环境变量

terminal:
```shell
export OPENAI_API_KEY="..."
```
py:
```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```
## 模块解释和应用

## 相关论文

· [MemPrompt: Memory-assisted Prompt Editing with User Feedback]( https://memprompt.com/)

