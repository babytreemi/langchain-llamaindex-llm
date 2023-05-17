# Connecting to a Feature Store

[什么是feature store?](https://www.tecton.ai/blog/what-is-a-feature-store/)
总的来说：特征存储是传统机器学习的一个概念，它确保输入模型的数据是最新的和相关的。

当考虑将LLM应用程序投入生产时，这个概念非常关键。为了个性化LLM应用程序，您可能希望将LLM与有关特定用户的最新信息结合起来。功能存储是保持数据新鲜的好方法，而LangChain提供了一种将数据与llm结合起来的简单方法。

在此笔记本中，我们将展示如何将提示模板连接到特征存储。**基本思想是从提示模板内部调用特征存储来检索值，然后将这些值格式化为提示。**

## [Feast](https://github.com/feast-dev/feast)

Feast是一个开源特征存储，用于管理和提供机器学习特征。它提供了一个统一的界面来定义特征，将特征数据从多个数据源加载到特征存储中，并将特征数据提供给托管的模型训练和在线推理服务。

`pip install feast`

**Load Feast Store**

```Python
from feast import FeatureStore

# You may need to update the path depending on where you stored it
# 注意需要初始化一个特征存储库（而不是随便新建一个文件夹放文件）具体看常见错误及解决方法.md
feast_repo_path = "path/to/my_new_feature_repo"
store = FeatureStore(repo_path=feast_repo_path)

```
设置一个自定义的`FeastPromptTemplate`。这个提示模板将接受一个驱动id，查找它们的统计信息，并将这些统计信息格式化为提示信息。
注意，这个提示模板的输入只有`driver_id`，因为这是唯一的用户定义部分(所有其他变量都在提示模板中查找)。

```Python
from langchain.prompts import PromptTemplate, StringPromptTemplate

template = """Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:"""
prompt = PromptTemplate.from_template(template)

``Python
class FeastPromptTemplate(StringPromptTemplate):
    
    def format(self, **kwargs) -> str:
        driver_id = kwargs.pop("driver_id")
        feature_vector = store.get_online_features(
            features=[
                'driver_hourly_stats:conv_rate',
                'driver_hourly_stats:acc_rate',
                'driver_hourly_stats:avg_daily_trips'
            ],
            entity_rows=[{"driver_id": driver_id}]
        ).to_dict()
        kwargs["conv_rate"] = feature_vector["conv_rate"][0]
        kwargs["acc_rate"] = feature_vector["acc_rate"][0]
        kwargs["avg_daily_trips"] = feature_vector["avg_daily_trips"][0]
        return prompt.format(**kwargs)
prompt_template = FeastPromptTemplate(input_variables=["driver_id"])
print(prompt_template.format(driver_id=1001))
```
**Use in a chain**
```Python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
chain = LLMChain(llm=ChatOpenAI(), prompt=prompt_template)
chain.run(1001)

# -> "Hi there! I wanted to update you on your current stats. Your acceptance rate is 0.055561766028404236 and your average daily trips are 936. While your conversation rate is currently 0.4745151400566101, I have no doubt that with a little extra effort, you'll be able to exceed that .5 mark! Keep up the great work! And remember, even chickens can't always cross the road, but they still give it their best shot."
```

## Tecton

下面的示例将展示使用Tecton的类似集成。Tecton是一个完全托管的功能平台，用于编排完整的机器学习功能生命周期，从转换到在线服务，具有企业级SLAs（service-level agreement）。

**Prerequisites** 

* Tecton Deployment (sign up at<tps://tecton.ai>)

* `TECTON_API_KEY `设置
  
**定义和加载Features**

```Python
user_transaction_metrics = FeatureService(
    name = "user_transaction_metrics",
    features = [user_transaction_counts]
)

import tecton

workspace = tecton.get_workspace("prod")
feature_service = workspace.get_feature_service("user_transaction_metrics")
```

**Prompts**

这里我们将设置一个自定义的 TectonPromptTemplate。**此提示模板将接受user_id，查找其统计信息，并将这些统计信息格式化为prompt。**
注意，这个提示模板的输入只有 `user_id` ，因为这是唯一的用户定义部分(所有其他变量都在提示模板中查找)。

```Python
from langchain.prompts import PromptTemplate, StringPromptTemplate
```

```Python
template = """Given the vendor's up to date transaction stats, write them a note based on the following rules:

1. If they had a transaction in the last day, write a short congratulations message on their recent sales
2. If no transaction in the last day, but they had a transaction in the last 30 days, playfully encourage them to sell more.
3. Always add a silly joke about chickens at the end

Here are the vendor's stats:
Number of Transactions Last Day: {transaction_count_1d}
Number of Transactions Last 30 Days: {transaction_count_30d}

Your response:"""
prompt = PromptTemplate.from_template(template)

class TectonPromptTemplate(StringPromptTemplate):
    
    def format(self, **kwargs) -> str:
        user_id = kwargs.pop("user_id")
        feature_vector = feature_service.get_online_features(join_keys={"user_id": user_id}).to_dict()
        kwargs["transaction_count_1d"] = feature_vector["user_transaction_counts.transaction_count_1d_1d"]
        kwargs["transaction_count_30d"] = feature_vector["user_transaction_counts.transaction_count_30d_1d"]
        return prompt.format(**kwargs)

prompt_template = TectonPromptTemplate(input_variables=["user_id"])
print(prompt_template.format(user_id="user_469998441571"))
```

**Use in a chain**
```Python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
chain = LLMChain(llm=ChatOpenAI(), prompt=prompt_template)
chain.run("user_469998441571")
```
## [Featureform](https://github.com/featureform/featureform) 
使用开源的企业级特性库`Featureform`来运行相同的示例。Featureform允许您使用像Spark这样的基础设施或在本地定义您的特性转换
 首先要根据仓库的README中的说明在Featureform中初始化transformation和features。

**Initialize Featureform**

 ```Python
import featureform as ff
client = ff.Client(host="demo.featureform.com")
```
**Prompts**
这里我们将设置一个自定义的FeatureformPromptTemplate。此提示模板将接收用户每次交易支付的平均金额。
注意，这个提示模板的输入只有avg_transaction，因为这是唯一的用户定义部分(所有其他变量都在提示模板中查找)。
```Python
from langchain.prompts import PromptTemplate, StringPromptTemplate
template = """Given the amount a user spends on average per transaction, let them know if they are a high roller. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the user's stats:
Average Amount per Transaction: ${avg_transcation}

Your response:"""
prompt = PromptTemplate.from_template(template)
class FeatureformPromptTemplate(StringPromptTemplate):
    
    def format(self, **kwargs) -> str:
        user_id = kwargs.pop("user_id")
        fpf = client.features([("avg_transactions", "quickstart")], {"user": user_id})
        return prompt.format(**kwargs)
prompt_template = FeatureformPrompTemplate(input_variables=["user_id"])
print(prompt_template.format(user_id="C1410926"))
```

**Use in a chain**
```Python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
chain = LLMChain(llm=ChatOpenAI(), prompt=prompt_template)
chain.run("C1410926")
```