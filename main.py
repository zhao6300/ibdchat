from app.workflow import RAGWorkflow
import os

os.environ["LANGCHAIN_TELEMETRY"] = "false"


def main():
    workflow = RAGWorkflow(
        llm_provider="basic",
        urls=[
            # "https://gitee.com/mindspore/mindspore/blob/master/README.md",
        ],
        # local_paths=["./README.md"]
        local_paths=["./智能打分系统.docx"],
    )
    # while True:
    #     question = input("请输入问题（输入exit退出）：")
    #     workflow.run(question)
    # print(workflow.mermaid_code)
    workflow.run("总结投行智能打分系统方案的特点和优势")
    # workflow.run("小鸡有几条腿？")


if __name__ == "__main__":
    main()
