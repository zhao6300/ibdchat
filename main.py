from app.workflow import RAGWorkflow

def main():
    workflow = RAGWorkflow(
        llm_provider="basic",
        urls=[
            # "https://gitee.com/mindspore/mindspore/blob/master/README.md",
        ],
        # local_paths=["./README.md"]
    )
    while True:
        question = input("请输入问题（输入exit退出）：")
        workflow.run(question)


if __name__ == "__main__":
    main() 
