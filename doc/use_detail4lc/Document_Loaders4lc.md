# Document Loaders

将语言模型与您自己的文本数据相结合的第一步是将数据加载到“Documents”中。

langchain使用 Document Loaders 实现这一步

## Transform loaders 转换加载器

这些tansform loaders 将数据从特定格式转换为 Document 格式。
例如，有用于 CSV 和 SQL 的转换器。大多数情况下，这些加载器从文件中输入数据，有时也从 URL 中输入数据。

许多这些转换器的主要驱动程序是 `Unstructured` python 包。这个包将多种类型的文件——文本、powerpoint、图像、html、pdf 等——转换为text数据。

[Unstructured-IO官方git仓库](https://github.com/Unstructured-IO/unstructured)
[Unstructured-IO官方文档](https://www.unstructured.io/)

针对不同格式文件的详细操作细节见：<https://python.langchain.com/en/latest/modules/indexes/document_loaders.html> 下面是一些常用的转换加载器：



### CoNLL-U
```python
rom langchain.document_loaders import CoNLLULoader
loader = CoNLLULoader("example_data/conllu.conllu")
document = loader.load()
document
# -> [Document(page_content='They buy and sell books.', metadata={'source': 'example_data/conllu.conllu'})]
```

### 复制粘贴

```python
from langchain.docstore.document import Document
text = "..... put the text you copy pasted here......"
doc = Document(page_content=text)```
```

**添加元数据**
```python
metadata = {"source": "internet", "date": "Friday"}
doc = Document(page_content=text, metadata=metadata)
```

### CSV

```python
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv')

data = loader.load()
print(data)
```

**自定义csv加载器**

```python
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']
})

data = loader.load()
print(data)
```

**指定一列来标识document来源**

使用`source_column`参数指定从每一行创建的document的来源。否则`file_path`将用作从 CSV 文件创建的所有文档的来源。

当使用从 CSV 文件加载的document用于使用source回答问题的chain时很有用。

```python
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', source_column="Team")

data = loader.load()
```
### Email

加载 `email (.eml)` & `Microsoft Outlook (.msg)` 文件.**

使用 `UnstructuredEmailLoader`

```python
# 要先 install unstructured
from langchain.document_loaders import UnstructuredEmailLoader
loader = UnstructuredEmailLoader('example_data/fake-email.eml')
data = loader.load()
data
```

**保留文本块信息**

`unstructured`会区分不同的文本块，langchian会默认把它拼接在一起，如果想保留文本块信息，可以指定`mode="elements"`参数

```python
loader = UnstructuredEmailLoader('example_data/fake-email.eml', mode="elements")
data = loader.load()
data[0]
```
**使用 OutlookMes​​sageLoader**

```python
from langchain.document_loaders import OutlookMessageLoader
loader = OutlookMessageLoader('example_data/fake-email.msg')
data = loader.load()
data[0]

# ->Document(page_content='This is a test email to experiment with the MS Outlook MSG Extractor\r\n\r\n\r\n-- \r\n\r\n\r\nKind regards\r\n\r\n\r\n\r\n\r\nBrian Zhou\r\n\r\n', metadata={'subject': 'Test for TIF files', 'sender': 'Brian Zhou <brizhou@gmail.com>', 'date': 'Mon, 18 Nov 2013 16:26:24 +0800'})
```
### 电子出版物

EPUB是一种使用“.epub”文件扩展名的电子书文件格式。

```python
#需要install pandocs
from langchain.document_loaders import UnstructuredEPubLoader
loader = UnstructuredEPubLoader("winter-sports.epub")
data = loader.load()
```
**保留文本块信息**

```python	
loader = UnstructuredEPubLoader("winter-sports.epub", mode="elements")
data = loader.load()
data[0]

# ->Document(page_content='The Project Gutenberg eBook of Winter Sports in\nSwitzerland, by E. F. Benson', lookup_str='', metadata={'source': 'winter-sports.epub', 'page_number': 1, 'category': 'Title'}, lookup_index=0)
```

其他支持的格式： 印象笔记（.enex）




## Public dataset or service loaders

## Proprietary dataset or service loaders


