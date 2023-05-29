# Indexes

索引指的是结构化文档的方法，以便llm能够最好地与文档进行交互。该模块包含用于**处理文档、不同类型索引的实用程序函数，以及在链中使用这些索引的示例。**

在链中使用索引的最常见方式是在“**retrieva**l”步骤中。这一步指的是接受用户的查询并返回最相关的文档。我们做出这样的区分是因为:

(1)index可以用于除retrive之外的其他用途

(2)retrive可以使用除index之外的其他逻辑来查找相关文档。因此有了一个“Retriver”接口的概念——-这是大多数chains使用的接口。

大多数时候，当我们谈论索引和检索时，我们谈论的是索引和检索非结构化数据(如文本文档)。要与结构化数据(SQL表等)或api交互，请参阅相应的用例部分，以获取相关功能的链接。目前，LangChain支持的主要索引和检索类型都集中在矢量(vector)数据库上。

We then provide a deep dive on the four main components. 四个重要的组成部分

**Document Loaders**

How to load documents from a variety of sources. 如何从各种来源加载文档

**Text Splitters**

An overview of the abstractions and implementions around splitting text.分割长文本为短文本

**VectorStores**

An overview of VectorStores and the many integrations LangChain provides.


**Retrievers**

An overview of Retrievers and the implementations LangChain provides.

## Getting Started

LangChain主要关注于构建索引，目标是将它们用作检索器。为了更好地理解这意味着什么，有必要强调一下基本的retriver接口是什么。LangChain中的`Basertriiever`类如下:

```python
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class BaseRetriever(ABC):
    @abstractmethod
	# 接受query 返回list[document]类型相关列表
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get texts relevant for a query.

        Args:
            query: string to find relevant texts for

        Returns:
            List of relevant documents
        """
```
支持自定义也可以使用langchain自带的策略

**Vectorstore**

默认情况下，LangChain使用`Chroma`作为向量存储来索引和搜索embeddings。macos用户`pip安装chroma遇到问题 见常用错误及解决方法。

文档问答四部曲：index retriever chain query

	Question answering over documents consists of four steps:

	1. Create an index

	2. Create a Retriever from that index

	3. Create a question answering chain

	4. Ask questions!

每个步骤都有多个子步骤和可能的配置。我们将主要关注(1)。我们将首先展示这样做的一行代码，然后分解解释实际发生的事情。 更多细节见：xx

首先，导入常用类。
```python 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
```
接下来，在通用设置中，让我们指定要使用的文档加载器,可以在这里下载[state_of_the_union.txt](https://github.com/hwchase17/langchain/blob/master/docs/modules/state_of_the_union.txt)

```PYTHON
from langchain.document_loaders import TextLoader
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
```
**One Line Index Creation**
```python
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# ask question
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
# -> " The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."

query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
# ->
# {'question': 'What did the president say about Ketanji Brown Jackson','answer': " The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\n",'sources': '../state_of_the_union.txt'}
```
`VectorstoreIndexCreator`返回的是`VectorStoreIndexWrapper`，它提供了上述`query`和`query_with_sources`功能。如果只是想直接访问vectorstore:

```python
index.vectorstore
#—> <langchain.vectorstores.chroma.Chroma at 0x119aa5940>
```
访问`VectorstoreRetriever`:
```python
index.vectorstore.as_retriever()
# ->VectorStoreRetriever(vectorstore=<langchain.vectorstores.chroma.Chroma object at 0x119aa5940>, search_kwargs={}) 
```

**How is this index getting created?**

当一个文档加载后，在`VectorstoreIndexCreator`中发生了三步: 

* Splitting documents into chunks 文档拆分

* Creating embeddings for each document 创建embedding

* Storing documents and embeddings in a vectorstore 存储

```python
# from langchain.document_loaders import TextLoader
# loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
documents = loader.load()
```
```python
# text spllitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documen(documents)                            
```

```python 
# select embedding
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```
```python
#  create the vectorstore to use as the index
from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
```

上面创建了index。然后我们要在retriever接口中公开该索引。

```python
retriever = db.as_retriever()
```

然后创建一个chain，使用上面的retriever来回答问题。

```python
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
```

`VectorstoreIndexCreator` 是所有这些逻辑的包装器,可以自由配置，比如可以配置如下：

```python
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, 
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
```

