import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

st.set_page_config(page_title="SWIN AI Coder", page_icon="🤖")
st.title("🤖 SWIN AI Coder")
st.write("Привет! Я твоя обученная нейросеть. Задай мне вопрос по коду или математике.")

@st.cache_resource
def load_model():
    # Базовая модель (мозг)
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # Ваши обученные мозги с Hugging Face
    adapter_name = "MAKSIM67771488/SWIN-AI-Coder"
    
    # Включаем 4-битное сжатие, чтобы влезть в память Streamlit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Загружаем базу
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Надеваем ваши обученные мозги (LoRA-адаптер)
    model = PeftModel.from_pretrained(base_model, adapter_name)
    
    return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# История чата
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода
if prompt := st.chat_input("Напиши свой вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Форматируем запрос
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        message_placeholder.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
