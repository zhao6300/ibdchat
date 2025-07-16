from app.workflow import RAGWorkflow

def main():
    workflow = RAGWorkflow(
        llm_provider="ollama",
        llm_model="llama2",
        urls=[
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
        ],
        local_paths=["./docs/example.md", "./docs/notes.txt"]
    )
    workflow.run("What are the types of agent memory?")

if __name__ == "__main__":
    main()