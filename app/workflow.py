import logging
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from typing import List, Literal, Optional, Dict
from typing_extensions import TypedDict, Annotated
from pprint import pprint
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_chroma import Chroma
from app.models import *
import os
from app.tools import *
from .prompt_template import *
import uuid

from app.models.chat_llm import get_llm
from dotenv import load_dotenv, find_dotenv
from pydantic_core import from_json

load_dotenv(find_dotenv())


class RouteQuery(BaseModel):
    thought: str = Field(...)
    next_step: str = Field(...)
    content: str = Field(...)


class GradeDocuments(BaseModel):
    thought: str = Field(...)
    binary_score: str = Field(...)


class GradeHallucinations(BaseModel):
    thought: str = Field(...)
    binary_score: str = Field(...)


class GradeAnswer(BaseModel):
    binary_score: str = Field(...)


class GenerateAnswer(BaseModel):
    """
    Structured output for a question-answering task, including a thought process and the final answer.
    """
    thought: str = Field(
        description="A brief reasoning process or internal monologue about how the answer was formulated, "
                    "or why the answer could not be found in the context."
    )
    answer: str = Field(
        description="The concise answer to the question, strictly based on the provided context. "
                    "It should be a maximum of three sentences. If the answer cannot be found in the context, "
                    "state: 'I cannot answer based on the provided context.'"
    )


class QueryCandidate(BaseModel):
    thought: str = Field(...)
    query: List[str] = Field(...)


class Plan(BaseModel):
    thought: str = Field(...)
    plans: List[str] = Field(
        description="A list of distinct, actionable steps or sub-tasks required to achieve a specific goal. Each item should be a clear, concise instruction.")


class HullucinationGrader(BaseModel):
    thought: str = Field(
        description="A brief reasoning process or internal monologue about how the answer was formulated, "
                    "or why the answer could not be found in the context."
    )
    binary_score: Literal["yes", "no"] = Field(
        description="A binary score indicating whether the LLM generation is grounded in the provided documents. "
                    "'yes' means it is grounded, 'no' means it is not."
    )


class AnswerGrader(BaseModel):
    thought: str = Field(
        description="A brief reasoning process or internal monologue about how the answer was formulated, "
                    "or why the answer could not be found in the context."
    )
    binary_score: Literal["yes", "no"] = Field(
        description="A binary score indicating whether the LLM generation is a useful answer to the question. "
                    "'yes' means it is useful, 'no' means it is not."
    )


class GraphState(TypedDict):
    thought: str
    question: str
    next_step: Optional[str]
    generation: str
    documents: List[Document]
    generation_count: int


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class ChromaDB:
    def __init__(self, collection_name: str = "test_db"):
        self.collection_name = collection_name

        model_type = os.getenv("EMBEDDING_MODEL_TYPE")
        model_name = os.getenv("EMBEDDING_MODEL_NAME")
        model_key = os.getenv("EMBEDDING_MODEL_API_KEY")
        modal_base_url = os.getenv("EMBEDDING_MODEL_BASE_URL")
        self.embedding = EmbeddingModel.get(model_type)(
            model_key, model_name, modal_base_url)
        self.db = Chroma(collection_name=collection_name,
                         embedding_function=self.embedding)

    def add_documents(self, documents: List[Document]):
        """Add documents to the ChromaDB collection."""
        return self.db.add_documents(documents)

    def retrieve(self, query: str, k: int = 2) -> List[Document]:
        """Retrieve documents based on a query."""
        return self.db.similarity_search(query, k=k)


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

        self._retriever = None
        self.db = ChromaDB(collection_name="rag-chroma")

    def build(self):
        docs: List[Document] = []
        # 加载网络文档
        for url in self.urls:
            docs.extend(WebBaseLoader(url).load())
        # 加载本地文档
        for path in self.local_paths:
            if path.endswith('.docx'):
                docs.extend(Docx2txtLoader(path).load())
            else:
                docs.extend(TextLoader(path).load())
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        corpus = splitter.split_documents(docs)
        if len(corpus) == 0:
            return None
        self.db.add_documents(corpus)
        return self.db


class Session(BaseModel):
    final_answer: str
    first_chat: bool = True
    thread: dict


class RAGWorkflow:
    def __init__(
        self,
        llm_provider: str = "basic",
        urls: Optional[List[str]] = None,
        local_paths: Optional[List[str]] = None
    ):
        self.llm = get_llm(llm_provider)
        self.analyzer = self._init_router()
        self.web_search_tool = tavily_tool
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
        self.mermaid_code = self.app.get_graph().draw_mermaid()
        self.sessions = {}
        self.max_generation_count = 3

    def _init_router(self):
        system = ANALYZER_PROMPT_TEMPLATE
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", "{question}"),
            ]
        )
        return route_prompt | self.llm

    def query_analysis(self, state: GraphState):
        res = self.analyzer.invoke({"question": state["question"]})
        route = RouteQuery.model_validate(from_json(res.content))
        return {**state, "next_step": route.next_step, "generation": route.content, "thought": route.thought}

    def choose(self, state: GraphState):
        if state["next_step"] == "vectorstore":
            return "vectorstore"
        elif state["next_step"] == "web_search":
            return "web_search"
        else:
            return "final_answer"

    def _plan(self, state: GraphState):
        system = PLANNER_PROMPT_TEMPLATE
        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        structured = self.llm.with_structured_output(Plan)
        res = plan_prompt | structured | self.llm
        return res.invoke({"question": state["question"]})

    def _init_retrieval_grader(self):

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DOCUMENTGRADER_PROMPT_TEMPLATE),
                ("human",
                 "Retrieved document: \n\n {documents} \n\n User question: {question}"),
            ]
        )

        return grade_prompt | self.llm

    def _init_rag_chain(self):

        prompt = PromptTemplate.from_template(GENERATOR_PROMPT_TEMPLATE)
        return prompt | self.llm

    def _init_hallucination_grader(self):
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", HALLUCINATOR_PROMPT_TEMPLATE),
                ("human",
                 "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        return hallucination_prompt | self.llm

    def _init_answer_grader(self):

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_GRADER_ROMPT_TEMPLATE),
                ("human",
                 "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        return answer_prompt | self.llm

    def _init_question_rewriter(self):

        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITER_PROMPT_TEMPLATE),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        return re_write_prompt | self.llm

    def _retrieve(self, state: GraphState) -> GraphState:
        docs = self.retriever.retrieve(state["question"])
        return {**state, "documents": docs}

    def _grade_documents(self, state: GraphState) -> GraphState:

        qs = state["question"]
        filtered = []
        for doc in state.get("documents", []):
            score_res = self.retrieval_grader.invoke(
                {"question": qs, "documents": doc.page_content})
            score = GradeDocuments.model_validate(from_json(score_res.content))
            if score.binary_score == "yes":
                filtered.append(doc)
        return {**state, "thought": score.thought, "documents": filtered}

    def _transform_query(self, state: GraphState) -> GraphState:

        out_res = self.question_rewriter.invoke(
            {"question": state["question"]})
        out = QueryCandidate.model_validate(from_json(out_res.content))
        return {**state, "question": out.query[0]}

    def _web_search(self, state: GraphState) -> GraphState:
        results = self.web_search_tool.invoke({"query": state["question"]})
        content = "\n".join(r["content"] for r in results)
        return {**state, "documents": [Document(page_content=content)]}

    def _generate(self, state: GraphState) -> GraphState:

        if self.max_generation_count > 0 and state.get("generation_count", 0) < self.max_generation_count:

            ctx = format_docs(state.get("documents", []))

            out_res = self.rag_chain.invoke(
                {"context": ctx, "question": state["question"]})
            pprint(out_res.content)
            out = GenerateAnswer.model_validate(from_json(out_res.content))

            hall_res = self.hallucination_grader.invoke({"documents": state.get(
                "documents", []), "generation": out.answer})
            hall = HullucinationGrader.model_validate(
                from_json(hall_res.content))
            if hall.binary_score == "yes":
                ans_res = self.answer_grader.invoke(
                    {"question": state["question"], "generation": out.answer})
                ans = AnswerGrader.model_validate(from_json(ans_res.content))
                if ans.binary_score == "yes":
                    return {**state, "thought": ans.thought,  "next_step": "final_answer", "generation": out.answer, 'generation_count': state.get("generation_count", 0) + 1}

            return {**state, "thought": ans.thought, "next_step": "query_write", 'generation_count': state.get("generation_count", 0) + 1, "generation": out.answer}

        return {**state, "next_step": "out_of_max_generation_count", 'generation_count': 0}

    def _grade_generation(self, state: GraphState) -> str:
        if state.get("final_answer") == "yes" or state.get("next_step") == "out_of_max_generation_count":
            return "end"
        elif state.get("final_answer") == "no":
            return "not useful"
        else:
            return "not supported"

    def _step(self, state: GraphState) -> GraphState:
        pass

    def _build_workflow(self):
        wf = StateGraph(GraphState)
        wf.add_node("planer", self._plan)
        wf.add_node("query_analysis", self.query_analysis)
        wf.add_node("step", self._step)
        wf.add_node("web_search", self._web_search)
        wf.add_node("retrieve", self._retrieve)
        wf.add_node("grade_documents", self._grade_documents)
        wf.add_node("query_rewrite", self._transform_query)
        wf.add_node("generate", self._generate)
        wf.add_edge(START, "query_analysis")
        wf.add_conditional_edges(
            "query_analysis",
            self.choose,
            {"web_search": "web_search", "vectorstore": "retrieve", "final_answer": END},
        )
        wf.add_edge("web_search", "generate")
        wf.add_edge("retrieve", "grade_documents")
        wf.add_conditional_edges(
            "grade_documents",
            lambda s: "generate" if s.get("documents") else "query_rewrite",
            {"query_rewrite": "query_rewrite", "generate": "generate"}
        )
        wf.add_edge("query_rewrite", "retrieve")
        wf.add_conditional_edges(
            "generate",
            self._grade_generation,
            {"end": END, "not useful": "query_rewrite",
                "not supported": "generate"}
        )
        return wf.compile()

    def run(self, question: str):
        session = self.sessions.get("user_id")
        if not session:
            thread_id = uuid.uuid4()
            thread = {"configurable": {"thread_id": thread_id}}
            session = Session(
                final_answer="no", thread=thread, first_chat=True)
            self.sessions["user_id"] = session
        thread = session.thread
        for output in self.app.stream({"question": question}, thread, stream_mode="updates", debug=True):
            if output.get("next_step") == "final_answer" or output.get("next_step") == "out_of_max_generation_count":
                pprint(output)
                break
