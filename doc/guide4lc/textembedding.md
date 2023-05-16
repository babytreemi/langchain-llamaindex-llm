# Text Embeddiing Models

Embedding 类是设计用于与embedding交互的类。有很多embedding提供商（OpenAI、Cohere、Hugging Face 等）—— 此类旨在为所有这些提供标准接口。

Embedding创建一段文本的矢量表示。这很有用，因为这意味着我们可以考虑向量空间中的文本，并执行语义搜索之类的操作，我们可以在向量空间中寻找最相似的文本片段。

LangChain 中的基础 `Embedding` 类公开了两个方法：`embed_documents`和`embed_query`。最大的区别在于这两种方法具有不同的接口：一种处理多个文档，而另一种处理单个文档。除此之外，将它们作为两种独立方法的另一个原因是一些嵌入提供商对文档（要搜索的）和查询（搜索查询本身）有不同的嵌入方法

## Aleph Alpha

注意使用前需要获取apikey
价格 0.03欧元/1000tokens

asymmetric embeddings 非对称

用于texts有不同的形式（ (e.g. a Document and a Query)）

```python
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding
document = "This is a content of the document"
query = "What is the contnt of the document?"
embeddings = AlephAlphaAsymmetricSemanticEmbedding()
doc_result = embeddings.embed_documents([document])
query_result = embeddings.embed_query(query)
```


symmetric embeddings 对称
```python
from langchain.embeddings import AlephAlphaSymmetricSemanticEmbedding
text = "This is a test text"
embeddings = AlephAlphaSymmetricSemanticEmbedding()
doc_result = embeddings.embed_documents([text])
query_result = embeddings.embed_query(text)
```

## AzureOpenAI

```python
# set the environment variables needed for openai package to know to reach out to azure
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="your-embeddings-deployment-name")
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```

## Cohere

```python

from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])

```

## Fake Embeddings

用于测试 pipelines

```python
from langchain.embeddings import FakeEmbeddings
embeddings = FakeEmbeddings(size=1352)
query_result = embeddings.embed_query("foo")
doc_results = embeddings.embed_documents(["foo"])
```

## Hugging Face Hub

运行代码发现 即使是在没有GPU的情况下（本人是macos），也可以使用这个模型

```python
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```

## InstructEmbeddings

加载 HuggingFace instruct Embeddings class.
```python
from langchain.embeddings import HuggingFaceInstructEmbeddings
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
# --> load INSTRUCTOR_Transformer
# --> max_seq_length  512
text = "This is a test document."
query_result = embeddings.embed_query(text)
```

## Jina

load  Jina Embedding class

```python
from langchain.embeddings import JinaEmbeddings
embeddings = JinaEmbeddings(jina_auth_token=jina_auth_token, model_name="ViT-B-32::openai")
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```
完整的jina模型列表：<https://jina.ai/>

## llama-cpp

`pip install llama-cpp-python`

```python
from langchain.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")
text = "This is a test document."
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])
```

## [openai](https://platform.openai.com/docs/guides/embeddings/use-cases)

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```

For 第一代生成模型（e.g. text-search-ada-doc-001/text-search-ada-query-001）

```python
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])
```

## SageMaker Endpoint Embeddings
加载SageMaker端点嵌入类。如果在SageMaker上托管自己的huging Face模型，则可以使用该类。

为了处理批量请求，你需要在自定义的inference.py脚本中调整predict_fn()函数中的返回行:

Change from

`return {"vectors": sentence_embeddings[0].tolist()}`

to:

`return {"vectors": sentence_embeddings.tolist()}`

`pip3 install langchain boto3`

```python
from typing import Dict, List
from langchain.embeddings import SagemakerEndpointEmbeddings
# 这个类是所有内容处理类的基类。
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
import json

# 继承自ContentHandlerBase
class ContentHandler(ContentHandlerBase):
	# 定义了ContentHandler类的两个类属性，分别表示内容类型和接受的内容类型都是JSON。
    content_type = "application/json"
    accepts = "application/json"

	# 接收两个参数：inputs（一个字符串列表）和model_kwargs（一个字典）。该方法返回一个字节串。
    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')
	
	# 它接收一个参数output（一个字节串），并返回一个嵌套的浮点数列表。
    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]

# 创建一个ContentHandler的实例。
content_handler = ContentHandler()


embeddings = SagemakerEndpointEmbeddings(
    # endpoint_name="endpoint-name", 
    # credentials_profile_name="credentials-profile-name", 
    endpoint_name="huggingface-pytorch-inference-2023-03-21-16-14-03-834", 
    region_name="us-east-1", 
    content_handler=content_handler
)
query_result = embeddings.embed_query("foo")
doc_results = embeddings.embed_documents(["foo"])
doc_results
```

## Self Hosted Embeddings


```python
from langchain.embeddings import (
    SelfHostedEmbeddings,
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)
import runhouse as rh

# For an on-demand A100 with GCP, Azure, or Lambda
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)

# For an on-demand A10G with AWS (no single A100s on AWS)
# gpu = rh.cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws')

# For an existing cluster
# gpu = rh.cluster(ips=['<ip of the cluster>'],
#                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
#                  name='my-cluster')


```

```python
embeddings = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
# embeddings = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)


text = "This is a test document."
query_result = embeddings.embed_query(text)
```

自定义load函数加载embedding model

```python

def get_pipeline():
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )  # Must be inside the function in notebooks

    model_id = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def inference_fn(pipeline, prompt):
    # Return last hidden state of the model
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]
embeddings = SelfHostedEmbeddings(
    model_load_fn=get_pipeline,
    hardware=gpu,
    model_reqs=["./", "torch", "transformers"],
    inference_fn=inference_fn,
)
query_result = embeddings.embed_query(text)
```

## Sentence Transformers Embeddings <https://www.sbert.net/>

`pip install -U sentence-transformers`

senencetransformers嵌入使用HuggingFaceEmbeddings集成来调用。我们还为更熟悉直接使用该包的用户添加了SentenceTransformerEmbeddings的别名。
SentenceTransformers是一个python包，可以生成文本和图像嵌入，起源于Sentence-BERT

```python
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text = "This is a test document."

query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text, "This is not a test document."])
```

## TensorflowHub

```python
from langchain.embeddings import TensorflowHubEmbeddings

embeddings = TensorflowHubEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)
doc_results = embeddings.embed_documents(["foo"])
doc_results
```

