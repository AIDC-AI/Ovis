import argparse
import os.path

import gradio as gr
from gradio.components import Textbox, Image

from ovis.serve.runner import RunnerArguments, OvisRunner


class Server:
    def __init__(self, runner: OvisRunner):
        self.runner = runner

    def __call__(self, image, text):
        if image is not None:
            response = self.runner.run(image, text)
            # output = f'prompt:\n{response["prompt"]}\n\noutput:\n{response["output"]}'
            output = response["output"]
        else:
            output = "No image uploaded. Please upload an image and retry."
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ovis Server')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--flagging_dir', type=str, default=os.path.expanduser('~/ovis-flagged'))
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.flagging_dir, exist_ok=True)
    runner_args = RunnerArguments(
        model_path=args.model_path
    )
    demo = gr.Interface(
        fn=Server(OvisRunner(runner_args)),
        inputs=[Image(type='pil', label='image'),
                Textbox(placeholder='Enter your text here...', label='prompt')],
        outputs=Textbox(),
        title=args.model_path.split('/')[-1],
        flagging_dir=args.flagging_dir
    )
    demo.launch(server_port=args.port)
