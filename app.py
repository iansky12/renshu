import os
import gradio as gr
from huggingface_hub import InferenceClient

# ------------------------------
# 1. AUTHENTICATION
# ------------------------------
MY_HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------
# 2. SYSTEM PROMPT (Personality)
# ------------------------------
REN_SHU_SYSTEM_PROMPT = """
You are Renshu (練習), a helpful Japanese language tutor.
Current Task: Explain Japanese concepts to the user.

Guidelines:
1. Explain primarily in English, but use Japanese examples.
2. If the user asks for a translation, provide: [Japanese Text] -> [Romaji] -> [English].
3. Be polite and encouraging.
"""

# ------------------------------
# 3. PROMPT BUILDER
# ------------------------------
def format_history_for_qwen(history, user_message):
    messages = [{"role": "system", "content": REN_SHU_SYSTEM_PROMPT}]
    for turn in history:
        if turn.get("role") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages

# ------------------------------
# 4. MAIN LOGIC
# ------------------------------
def respond(
    message,
    history: list[dict[str, str]],
    max_tokens,
    temperature,
    top_p,
):
    if not MY_HF_TOKEN:
        yield "⚠️ **Setup Error:** Token missing! Go to 'Settings > Variables and secrets' and add 'HF_TOKEN'."
        return

    try:
        # BEST FREE OPTION: Qwen 2.5 72B (Official)
        # Note: If this fails with 503, change to "Qwen/Qwen2.5-7B-Instruct"
        model_id = "Qwen/Qwen2.5-72B-Instruct"
        
        client = InferenceClient(
            token=MY_HF_TOKEN,
            model=model_id,
        )

        messages = format_history_for_qwen(history, message)

        yield f"🇯🇵 Renshu is thinking (using {model_id})..."

        stream = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )

        partial_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                partial_response += token
                yield partial_response

    except Exception as e:
        yield f"❌ Error: {str(e)}\n\nTry switching the model to 'Qwen/Qwen2.5-7B-Instruct' in the code if the 72B version is too busy."

# ------------------------------
# 5. UI SETUP
# ------------------------------
custom_css = """
body { font-family: 'Noto Sans JP', sans-serif; }
.message-wrap { font-size: 1.1em !important; line-height: 1.6 !important; }
"""

chat_interface = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Slider(128, 1024, value=512, step=32, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    examples=[
        ["How do I say 'I've been studying Japanese for 3 years'?"],
        ["What is the difference between は (wa) and が (ga)?"],
    ],
    cache_examples=False
)

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Renshu AI") as demo:
    gr.Markdown("# 🎌 Renshu (練習) - AI Tutor")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("""
            ### 👋 Welcome!
            **Renshu** uses **Qwen 2.5 (72B)** to teach you Japanese.
            """)
        with gr.Column(scale=3):
            chat_interface.render()

if __name__ == "__main__":
    demo.launch()