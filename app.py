import gradio as gr
import pandas as pd
import pickle
import sklearn
import gradio as gr
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if gr.NO_RELOAD:
    model = pickle.load(open('./model/model.sav', 'rb'))
    columns = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Acoustic_Level', 'Temperature']

def fault_predictor(v_x, v_y, v_z, al, temp):
    x = pd.DataFrame([[v_x, v_y, v_z, al, temp]], columns = columns)
    result = str(model.predict(x))[2:-2]

    if result == "Normal": 
        return f"<h1 style = 'text-align: center;'>Machine Status: <span style='color: #10b981;'>{result}</span></h1>"
    else:
        return f"<h1 style = 'text-align: center;'>Machine Status: <span style='color: #f52727;'>{result}</span></h1>"

def visualizer(v_x, v_y, v_z, al, temp, history):
    vals = [round(float(v_x), 2), round(float(v_y), 2), round(float(v_z), 2), round(float(al), 2), round(float(temp), 2)]
    history = history + [vals]

    parameters = ["X Vibration", "Y Vibration", "Z Vibration", "Acoustic Level", "Temperature"]
    fig = make_subplots(
        rows = 5,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing=0.04,
    )

    runs = list(range(1, len(history) + 1))

    for i, name in enumerate(parameters, start=1):
        fig.add_trace(
            go.Scatter(
                x = runs,
                y = [row[i - 1] for row in history],
                mode = "lines+markers",
                name = name,
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=i, col=1)

    fig.update_xaxes(title_text="Run number", row = len(parameters), col = 1)
    if len(runs) <= 20:
        fig.update_xaxes(dtick=1)  # whole-number ticks (auto-spaced beyond 20 runs)

    fig.update_layout(
        height=800,
        margin=dict(l=60, r=20, t=20, b=50),
        hovermode="x unified",  # hover shows all channels for that run
        template="plotly_white",
    )

    return fig, history

with gr.Blocks() as demo:
    fn = fault_predictor

    title = gr.Markdown("<h1 style = 'text-align: center; font-size: 40px'>Rotating Industrial Equipment Fault Prediction</h1>")

    with gr.Tabs():
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    submit_instructions = gr.Markdown("##### To assess your equipment's condition, enter the following values and press submit:")

                    x = gr.Textbox(label = 'Vibration - X (Normalized)', info = 'Between 0 - 1.3') 
                    y = gr.Textbox(label = 'Vibration - Y (Normalized)', info = 'Between 0 - 1.3') 
                    z = gr.Textbox(label = 'Vibration - Z (Normalized)', info = 'Between 0 - 1.3') 
                    acoustic = gr.Textbox(label = 'Acoustic Level - dB', info = 'Between 10 - 110') 
                    temp = gr.Textbox(label = 'Temperature - ºC)', info = 'Between 50 - 110') # units are extrapolated

                    with gr.Row():
                        submit = gr.Button("Submit", variant="primary")
                        clear = gr.ClearButton([x, y, z, acoustic, temp])
                
                with gr.Column():
                    example_instructions = gr.Markdown("##### Or, pre-populate an example by clicking a row of values:")
                    with gr.Accordion("Show/hide examples"):
                        examples = gr.Examples(examples = "examples", inputs = [x, y, z, acoustic, temp], examples_per_page = 9)
                    
            with gr.Row():
                output = gr.Markdown("<h1 style = 'text-align: center;'>Machine Status: Undetermined</h1>")
            
            submit.click(fn = fault_predictor, inputs = [x, y, z, acoustic, temp], outputs = output, api_name = "predict")

        with gr.Tab("Vizualize"):
            history = gr.BrowserState([])
            plot = gr.Plot()

            submit.click(fn = visualizer, inputs = [x, y, z, acoustic, temp, history], outputs = [plot, history])
    
demo.launch(theme = gr.themes.Ocean())