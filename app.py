import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Настройка страницы
st.set_page_config(page_title="SWIN AI", page_icon="🤖")
st.title("🤖 SWIN AI Coder")
st.write("Привет! Я твоя нейросеть. Задай мне любой вопрос по коду или математике.")

# Загрузка модели (кэшируется, чтобы не грузить каждый раз заново)
@st.cache_resource
def load_model():
    # Используем компактную модель Qwen 0.5B, чтобы она точно поместилась в память
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return model, tokenizer

# Инициализация модели
model, tokenizer = load_model()

# Хранение истории сообщений в сессии
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода
if prompt := st.chat_input("Напиши свой вопрос..."):
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Форматируем запрос для модели Qwen
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        # Генерация без градиентов (экономит память)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        message_placeholder.markdown(response)
        
    # Сохраняем ответ в историю
    st.session_state.messages.append({"role": "assistant", "content": response})
