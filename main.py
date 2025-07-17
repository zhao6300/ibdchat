from app.workflow import RAGWorkflow


def main():
    workflow = RAGWorkflow(
        llm_provider="basic",
        urls=[
            "https://gitee.com/mindspore/mindspore/blob/master/README.md",
        ],
        local_paths=["./README.md"]
    )
    workflow.run("What are the types of agent memory?")


if __name__ == "__main__":
    main()
