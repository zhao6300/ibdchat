from langgraph.graph import END, StateGraph, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import List, Literal, Optional, Dict
from typing_extensions import TypedDict
from pprint import pprint
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain.document_loaders import TextLoader  # 支持本地文本加载
from langchain_community.vectorstores import Chroma
from app.models import *
import os


from app.models.chat_llm import get_llm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...)


class GradeDocuments(BaseModel):
    binary_score: str = Field(...)


class GradeHallucinations(BaseModel):
    binary_score: str = Field(...)


class GradeAnswer(BaseModel):
    binary_score: str = Field(...)


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class DocumentVectorizer:
    def __init__(
        self,
        urls: Optional[List[str]] = None,
        local_paths: Optional[List[str]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 0
    ):
        self.urls = urls or []
        self.local_paths = local_paths or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        model_type = os.getenv("EMBEDDING_MODEL_TYPE")
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        model_key = os.getenv("EMBEDDING_MODEL_API_KEY")
        modal_base_url = os.getenv("EMBEDDING_MODEL_BASE_URL")
        self.embedding = EmbeddingModel.get(model_type)(
            model_name, model_name, modal_base_url)
        self._retriever = None

    def build(self):
        docs: List[Document] = []
        # 加载网络文档
        for url in self.urls:
            docs.extend(WebBaseLoader(url).load())
        # 加载本地文档
        for path in self.local_paths:
            docs.extend(TextLoader(path).load())
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        corpus = splitter.split_documents(docs)
        store = Chroma.from_documents(
            documents=corpus,
            collection_name="rag-chroma",
            embedding=self.embedding
        )
        self._retriever = store.as_retriever()
        return self._retriever


class RAGWorkflow:
    def __init__(
        self,
        llm_provider: str = "basic",
        urls: Optional[List[str]] = None,
        local_paths: Optional[List[str]] = None
    ):
        self.llm = get_llm(llm_provider)
        self.question_router = self._init_router()
        self.web_search_tool = TavilySearchResults(k=3)
        vectorizer = DocumentVectorizer(
            urls=urls,
            local_paths=local_paths
        )
        self.retriever = vectorizer.build()
        self.retrieval_grader = self._init_retrieval_grader()
        self.rag_chain = self._init_rag_chain()
        self.hallucination_grader = self._init_hallucination_grader()
        self.answer_grader = self._init_answer_grader()
        self.question_rewriter = self._init_question_rewriter()
        self.app = self._build_workflow()

    def _init_router(self):
        system = """You are an expert at routing a user question to a vectorstore or web search.
                The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        structured = self.llm.with_structured_output(RouteQuery)
        return route_prompt | structured

    def _init_retrieval_grader(self):

        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human",
                 "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        structured = self.llm.with_structured_output(GradeDocuments)
        return grade_prompt | structured

    def _init_rag_chain(self):

        prompt_str = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context} 
        Answer:"""

        prompt = PromptTemplate.from_template(prompt_str)
        return prompt | self.llm | StrOutputParser()

    def _init_hallucination_grader(self):

        structured = self.llm.with_structured_output(GradeHallucinations)
        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human",
                 "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        return hallucination_prompt | structured

    def _init_answer_grader(self):

        structured = self.llm.with_structured_output(GradeAnswer)
        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human",
                 "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        return answer_prompt | structured

    def _init_question_rewriter(self):

        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        return re_write_prompt | self.llm | StrOutputParser()

    def _retrieve(self, state: GraphState) -> GraphState:
        docs = self.retriever.invoke(state["question"])
        return {**state, "documents": docs}

    def _grade_documents(self, state: GraphState) -> GraphState:

        qs = state["question"]
        filtered = []
        for doc in state.get("documents", []):
            score = self.retrieval_grader.invoke(
                {"question": qs, "document": doc.page_content})
            if score.binary_score == "yes":
                filtered.append(doc)
        return {**state, "documents": filtered}

    def _transform_query(self, state: GraphState) -> GraphState:

        new_q = self.question_rewriter.invoke({"question": state["question"]})
        return {**state, "question": new_q}

    def _web_search(self, state: GraphState) -> GraphState:
        results = self.web_search_tool.invoke({"query": state["question"]})
        content = "\n".join(r["content"] for r in results)
        return {**state, "documents": [Document(page_content=content)]}

    def _generate(self, state: GraphState) -> GraphState:
        ctx = format_docs(state.get("documents", []))
        out = self.rag_chain.invoke(
            {"context": ctx, "question": state["question"]})
        return {**state, "generation": out}

    def _grade_generation(self, state: GraphState) -> str:
        hall = self.hallucination_grader.invoke({"documents": state.get(
            "documents", []), "generation": state.get("generation", "")})
        if hall.binary_score == "yes":
            ans = self.answer_grader.invoke(
                {"question": state["question"], "generation": state["generation"]})
            return "useful" if ans.binary_score == "yes" else "not useful"
        return "not supported"

    def _build_workflow(self):
        wf = StateGraph(GraphState)
        wf.add_node("web_search", self._web_search)
        wf.add_node("retrieve", self._retrieve)
        wf.add_node("grade_documents", self._grade_documents)
        wf.add_node("transform_query", self._transform_query)
        wf.add_node("generate", self._generate)
        wf.add_conditional_edges(
            START,
            lambda s: self.question_router.invoke(
                {"question": s["question"]}).datasource,
            {"web_search": "web_search", "vectorstore": "retrieve"}
        )
        wf.add_edge("web_search", "generate")
        wf.add_edge("retrieve", "grade_documents")
        wf.add_conditional_edges(
            "grade_documents",
            lambda s: "generate" if s.get("documents") else "transform_query",
            {"transform_query": "transform_query", "generate": "generate"}
        )
        wf.add_edge("transform_query", "retrieve")
        wf.add_conditional_edges(
            "generate",
            self._grade_generation,
            {"useful": END, "not useful": "transform_query",
                "not supported": "generate"}
        )
        return wf.compile()

    def run(self, question: str):
        for output in self.app.stream({"question": question}):
            pprint(output)
        print(output.get("generation", ""))
