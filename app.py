import gradio as gr
import pandas as pd
import pickle
import sklearn
import gradio as gr

if gr.NO_RELOAD:
    model = pickle.load(open('./model/model.sav', 'rb'))
    columns = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Acoustic_Level', 'Temperature']

def fault_predictor(v_x, v_y, v_z, al, temp):
    x = pd.DataFrame([[v_x, v_y, v_z, al, temp]], columns = columns)
    result = str(model.predict(x))[2:-2]

    if result == "Normal":
        return f"## Machine Status: <span style='color: #10b981;'>{result}</span>"
    else:
        return f"## Machine Status: <span style='color: #f52727;'>{result}</span>"

with gr.Blocks() as demo:
    fn = fault_predictor

    title = gr.Markdown("# Rotating Industrial Equipment Fault Prediction")

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
        output = gr.Markdown()

    submit.click(fn = fault_predictor, inputs = [x, y, z, acoustic, temp], outputs = output, api_name = "predict")

demo.launch(theme = gr.themes.Ocean())