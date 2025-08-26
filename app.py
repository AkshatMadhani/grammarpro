import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = 'vennify/t5-base-grammar-correction'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=False)
model = model.to(torch_device)

def correct_grammar(input_text, num_return_sequences=1):
    batch = tokenizer(
        [input_text],
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors="pt"
    ).to(torch_device)

    outputs = model.generate(
        **batch,
        max_length=64,
        num_beams=4,
        num_return_sequences=num_return_sequences,
        temperature=1.5
    )

    corrected_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return corrected_sentences

st.title('Grammar Error Correction')

input_text = st.text_area("Enter text to correct:")

if st.button('Correct Grammar'):
    corrected = correct_grammar(input_text)
    st.subheader("Corrected Text:")
    st.write(corrected[0])
