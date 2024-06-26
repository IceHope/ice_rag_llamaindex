import gradio as gr

from rag_init import RagInit
from vector_store_client import VectorStoreClient

RagInit()

db_client = VectorStoreClient(collection_name="rag_lla")


def upload_files(files):
    file_names = [file.name for file in files]
    db_client.add_files(file_names)
    return f"上传成功, 缓存的本地地址 :\n {', '.join(file_names)}"


def clear_files():
    return "请上传文件"


def user(user_message, history):
    return "", history + [[user_message, None]]


def retry_user(history):
    if history:
        last_query = history[-1][0]
        return history + [[last_query, None]]
    else:
        return history


def bot(history):
    query = history[-1][0]
    response = db_client.chat(query)
    print("\nbot query:", query, "\nresponse:", response)
    return response


def stream_bot(history):
    query = history[-1][0]
    response = ""
    response_gen = db_client.stream_chat(query)
    for gen in response_gen:
        response += gen
        history[-1][1] = response
        yield history

    print("\nstream_bot query:", query, "\nresponse:", response)


def undo(history):
    if history:
        history.pop()
    return history


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Rag Using Llamaindex</h1>""")

        with gr.Row():
            with gr.Column(scale=1):
                files_upload = gr.Files(
                    label="上传文件,支持.pdf格式", file_types=["pdf"], scale=2)
                upload_result = gr.Textbox(
                    show_label=False, placeholder="请上传文件", scale=4)
            with gr.Column(scale=4):
                chatbot = gr.Chatbot()

                user_input = gr.Textbox(show_label=False, placeholder="输入框...")
                with gr.Row():
                    retry_btn = gr.Button("Retry")
                    undo_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Clear")

        user_input.submit(user, inputs=[user_input, chatbot], outputs=[user_input, chatbot], queue=False
                          ).then(stream_bot, chatbot, chatbot)

        retry_btn.click(retry_user, inputs=chatbot, outputs=chatbot, queue=False
                        ).then(stream_bot, chatbot, chatbot)

        undo_btn.click(undo, inputs=chatbot, outputs=chatbot, queue=False)

        clear_btn.click(lambda: None, None, chatbot, queue=False)

        files_upload.upload(upload_files, inputs=files_upload, outputs=upload_result)
        files_upload.clear(clear_files, outputs=upload_result)

        demo.queue().launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=8060)


if __name__ == "__main__":
    main()
