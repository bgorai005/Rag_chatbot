# File: generator.py
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_core.output_parsers import StrOutputParser
import re
from typing import List
from config import GROQ_API_KEY, LLM_MODEL

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

class Generator:
    def __init__(self):
        self.qa_prompt = PromptTemplate(
            template="""Based on the following context from research papers, answer the question grounded in the text. Cite chunks if possible.

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        self.summary_prompt = PromptTemplate(
            template="""Summarize the key points of this research paper:
{text}""",
            input_variables=["text"]
        )
        self.highlights_prompt = PromptTemplate(
            template="""Extract 5 key highlights or bullet points from this paper:
{text}""",
            input_variables=["text"]
        )
        self.compare_prompt = PromptTemplate(
            template="""Compare and contrast the following papers based on their summaries:
{combined}

Comparison:""",
            input_variables=["combined"]
        )

    def create_qa_chain(self, retriever):
        """Create RetrievalQA chain."""
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )

    def qa_pipeline(self, question: str, context: List[str]) -> str:
        """Grounded Q&A using context (fallback to direct chain if retriever not passed)."""
        context_str = "\n\n".join([doc['chunk'] for doc in context])
        chain = LLMChain(llm=llm, prompt=self.qa_prompt, output_parser=StrOutputParser())
        answer = chain.invoke({"context": context_str, "question": question})['text']
        return answer

    def summarize_paper(self, text: str, section_wise: bool = False) -> str:
        """Generate summary using LangChain chain."""
        if section_wise:
            # Simplified section split
            sections = re.split(r'\n[A-Z][a-z]+\n', text)
            summaries = []
            for sec in sections:
                if sec.strip():
                    chain = LLMChain(llm=llm, prompt=PromptTemplate(template="Summarize this section: {sec}", input_variables=["sec"]), output_parser=StrOutputParser())
                    summ = chain.invoke({"sec": sec[:2000]})['text']
                    summaries.append(summ)
            return "\n\n".join(summaries)
        else:
            chain = LLMChain(llm=llm, prompt=self.summary_prompt, output_parser=StrOutputParser())
            return chain.invoke({"text": text[:4000]})['text']

    def highlights(self, text: str) -> str:
        """Extract highlights using chain."""
        chain = LLMChain(llm=llm, prompt=self.highlights_prompt, output_parser=StrOutputParser())
        return chain.invoke({"text": text[:4000]})['text']

    def compare_papers(self, summaries: List[str]) -> str:
        """Compare and contrast using chain."""
        if len(summaries) < 2:
            return "Need at least 2 papers."
        combined = "\n\n---\n\n".join([f"Paper {i+1}: {summ}" for i, summ in enumerate(summaries)])
        chain = LLMChain(llm=llm, prompt=self.compare_prompt, output_parser=StrOutputParser())
        return chain.invoke({"combined": combined})['text']