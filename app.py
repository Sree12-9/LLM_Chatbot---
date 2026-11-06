
import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# APP CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Smart AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
ADAPTER_FOLDER = BASE_DIR / "qwen_chatbot_finetuned"


# -----------------------------
# FIND LORA ADAPTER
# -----------------------------

def find_adapter():

    possible_folders = [ADAPTER_FOLDER]

    # Also check checkpoint folders
    if ADAPTER_FOLDER.exists():
        possible_folders.extend(
            sorted(
                ADAPTER_FOLDER.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1])
                if p.name.split("-")[-1].isdigit() else -1,
                reverse=True
            )
        )

    for folder in possible_folders:

        config_file = folder / "adapter_config.json"

        weights_file = (
            folder / "adapter_model.safetensors"
        )

        weights_bin = folder / "adapter_model.bin"

        if config_file.exists() and (
            weights_file.exists() or weights_bin.exists()
        ):
            return folder

    raise FileNotFoundError(
        f"No valid LoRA adapter found in: {ADAPTER_FOLDER}\n"
        "Expected adapter_config.json and adapter_model.safetensors "
        "or adapter_model.bin."
    )


# -----------------------------
# LOAD FINE-TUNED MODEL
# -----------------------------

@st.cache_resource
def load_model():

    adapter_path = find_adapter()

    st.info(f"Loading adapter: {adapter_path.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select device and data type
    if torch.cuda.is_available():

        device = "cuda"
        dtype = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto"
        )

    else:

        device = "cpu"

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )

        base_model.to(device)

    # Load trained LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path)
    )

    model.eval()
    model.config.use_cache = True

    return tokenizer, model


# -----------------------------
# GENERATE RESPONSE
# -----------------------------

def generate_response(chat_messages, tokenizer, model):

    # Convert conversation into Qwen chat format
    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Convert text into tensors
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False
    )

    # Move tensors to model device
    device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate answer
    with torch.inference_mode():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract only newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    new_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# -----------------------------
# STREAMLIT USER INTERFACE
# -----------------------------

st.title("🤖 Smart AI Chatbot")

st.caption(
    "Powered by Qwen2.5-0.5B-Instruct | "
    "UltraChat Dataset | SFT + LoRA Fine-Tuning"
)

# Load model
try:

    with st.spinner("Loading fine-tuned model..."):

        tokenizer, model = load_model()

    st.success("Fine-tuned model loaded successfully!")

except Exception as e:

    st.error("Model loading failed!")
    st.exception(e)

    st.stop()


# -----------------------------
# CHAT HISTORY
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Clear conversation
if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []
    st.rerun()


st.sidebar.title("About")

st.sidebar.write(
    "This chatbot uses a pretrained Qwen language model "
    "fine-tuned using Supervised Fine-Tuning (SFT) "
    "and LoRA."
)


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# USER INPUT
# -----------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = generate_response(
                    st.session_state.messages,
                    tokenizer,
                    model
                )

                if not answer:
                    answer = "I could not generate a response. Please try again."

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error("Response generation failed!")

                # Display the actual error and traceback
                st.exception(e)
