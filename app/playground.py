import argparse
import sys
import os
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.generate import load_model_for_inference, generate
from tokenizer import LLMTokenizer


def build_playground(checkpoint_path, model_config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model (device: {device})...")
    model = load_model_for_inference(checkpoint_path, model_config_path, device)
    tokenizer = LLMTokenizer()
    print(f"Model ready: {model.num_parameters():,} parameters")

    def respond(message, history, temperature, top_k, top_p, max_new_tokens):
        conversation = ""
        for user_msg, bot_msg in history:
            conversation += f"<|user|>{user_msg}<|assistant|>{bot_msg}"
        conversation += f"<|user|>{message}<|assistant|>"

        full_output, response_text = generate(
            model, tokenizer, conversation,
            max_new_tokens=int(max_new_tokens), temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
            device=device,
        )
        response = response_text.strip()
        history.append((message, response))
        return "", history

    with gr.Blocks(title="LLM Playground") as demo:
        gr.Markdown("# LLM Playground")
        gr.Markdown(f"Trained model: **{model.num_parameters():,} parameters** - running on `{device}`")

        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Message", placeholder="Type your message here...")

        with gr.Accordion("Generation Parameters (Advanced)", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature",
                                          info="Higher = more random/diverse")
                top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k",
                                    info="0 = disable top-k filtering")
                top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p (nucleus)",
                                    info="1.0 = disable top-p filtering")
                max_new_tokens = gr.Slider(10, 300, value=150, step=10, label="Max new tokens")

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Conversation")

        submit_btn.click(respond, [msg, chatbot, temperature, top_k, top_p, max_new_tokens],
                          [msg, chatbot])
        msg.submit(respond, [msg, chatbot, temperature, top_k, top_p, max_new_tokens],
                   [msg, chatbot])
        clear_btn.click(lambda: (None, []), None, [msg, chatbot])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt")
    parser.add_argument("--model_config", type=str, default="configs/model_small.yaml")
    args = parser.parse_args()

    app = build_playground(args.checkpoint, args.model_config)
    app.launch(share=True)