import torch
import gradio as gr
from model import PINN
import numpy as np

def load_model():
    model = PINN(layers=[2, 32, 32, 32, 1])
    model.load_state_dict(torch.load("outputs/trained_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(x_val, t_val):
    input_tensor = torch.tensor([[x_val, t_val]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(0, 1, step=0.01, label="x"), gr.Slider(0, 1, step=0.01, label="t")],
    outputs=gr.Number(label="u(x,t) Prediction")
)

if __name__ == "__main__":
    demo.launch()
