# langchain-llamaindex-llm
因为这两个库迭代太快➕看不懂大佬们写的代码➕英语不好➕踩坑太多➕环境配置总是冲突，所以自己造一个简单的仓库方便自己造福和我一样的小可怜

## 常用链接
langchain集成的列表：https://langchain.com/integrations.html
langchain Pythonrepo：https://github.com/hwchase17/langchain
langchain doc：https://python.langchain.com/en/latest/index.html
## 常用命令

### 1. install
`pip install langchain` 
or 
`conda install langchain -c conda-forge`

这里有个坑（包括安装其他的一些库时），多次遇到在conda环境中使用pip命令会找不到，

解决办法：1. `conda install pip`，然后再用`pip`安装langchain 2.`python -m pip install langchain`

