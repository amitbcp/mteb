import gradio as gr
TOP_N = 10
demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(placeholder="Enter your query text here...", label="Query Text"),
        gr.Image(label="Query Image", type="numpy")
    ],
    outputs=gr.Gallery(label=f"Retrieved Images (Top {TOP_N})", columns=3),
    # examples=load_examples(),
    title="Instance-Driven Multi-modal Retrieval (IDMR) Demo",
    description="Enter a query text or upload an image to retrieve relevant images from the library. You can click on the examples below to try them out."
)
