from img2art_search.models.predict import predict
import gradio as gr

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Gallery(label="Most similar images", height=256 * 3),
    live=True,
)
interface.launch()