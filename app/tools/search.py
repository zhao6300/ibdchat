from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field
from langchain_community.utilities import GoogleSearchAPIWrapper
import asyncio
import pprint
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
import langchain_community.document_loaders.async_html
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Any, Optional, Type
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from app.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool


from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchRun
import argparse
import json
from dotenv import load_dotenv
load_dotenv()


# Initialize Tavily search tool with logging
LoggedTavilySearch = create_logged_tool(TavilySearchResults)
tavily_tool = LoggedTavilySearch(
    name="tavily_search", max_results=TAVILY_MAX_RESULTS)


api_wrapper = DuckDuckGoSearchAPIWrapper(
    time=None, max_results=3, backend="lite")
html2text = Html2TextTransformer()


class WebSearchInput(BaseModel):
    query: str = Field(..., description=f"search query")


def duckduckgo_search(query: str, top_k: int):

    results = api_wrapper.results(query, max_results=top_k)
    # print(results)
    urls = [res["link"] for res in results]
    # loader = AsyncChromiumLoader(urls)
    # AsyncHtmlLoader这个效果不是那么好, 还是要换成AsyncChromiumLoader
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    for doc in docs:
        if doc.page_content == '':
            doc.page_content = doc.metadata.get('description', '')
    # print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n{docs}")
    # docs_transformed = self.bs_transformer.transform_documents(docs, unwanted_tags=['li','a'],tags_to_extract=["p",'div'])
    docs_transformed = html2text.transform_documents(docs)
    # print(f"################################\n{docs_transformed}")
    # print(res)
    # print(docs_transformed[0].page_content)
    # 这里加上title是不是好一点
    search_contents = []
    for i, doc in enumerate(docs_transformed):
        title_content = results[i]["title"]
        search_contents.append(
            f">>>>>>>>>>>>>>>>>>>>以下是标题为<h1>{title_content}</h1>的网页内容\n{doc.page_content}\n<<<<<<<<<<<<<<<<<以上是标题为<h1>{title_content}</h1>的网页内容\n")
    return "\n\n".join([doc for doc in search_contents]), docs_transformed
    # return ", ".join([res["snippet"] for res in results])


web_search_tool = StructuredTool.from_function(
    func=duckduckgo_search,
    name="duckduckgo_search",
    description="Search infomation on internet. Useful for when the context can not answer the question. Input should be a search query.",
    args_schema=WebSearchInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

tools = [web_search_tool]
functions = [convert_to_openai_function(t) for t in tools]
print(f"functions:{functions}", flush=True)

search_tools = []
college_tool = {"type": "function", "function": functions[0]}
search_tools.append(college_tool)

if __name__ == "__main__":
    result = duckduckgo_search("985大学有哪些?", top_k=3)
    print(result)
