from config import PATH_PROJECT_PROMPT


def get_rag_query_prompt():
    path = PATH_PROJECT_PROMPT + "/rag_query_prompt.txt"
    with open(path, "r") as f:
        prompt = f.read()
        return prompt


if __name__ == "__main__":
    print(get_rag_query_prompt())
