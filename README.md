# ⚖️ Legal-QA Fine-Tuned Model (Mistral-7B)
A **domain-adapted Legal AI Assistant** built by **fine-tuning Mistral-7B-Instruct** on a **Legal Q&A dataset** using **LoRA (Low-Rank Adaptation)**.  
This project demonstrates how to adapt large language models for specialized domains like **law, compliance, and legal research** on limited compute (Colab T4 GPU).

---

## 📌 Project Overview
- **Base Model**: [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  
- **Dataset**: Custom Legal **Question → Answer** pairs, formatted into instruction style.  
- **Training Method**: **LoRA fine-tuning** with Hugging Face `transformers` + `peft`.  
- **Hardware**: Google Colab T4 GPU (16GB VRAM).  
- **Goal**: Improve performance of Mistral on **domain-specific legal queries** while being cost-efficient.  

---

## ⚡ Features
✅ Domain-specific **Legal Q&A understanding**  
✅ Efficient training using **LoRA adapters**  
✅ Works on **consumer-grade GPUs (T4)**  
✅ Easily extendable to other datasets (Medical, Finance, etc.)  

---

## 📂 Repository Structure
├── legalfinetuningllm.ipynb # Colab notebook with full training pipeline
├── ipc_qa.json/ # Legal Q&A dataset (JSON/CSV)
├── README.md # Project documentation


---

## 🚀 Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/legal-qa-mistral.git
cd legal-qa-mistral
2️⃣ Install dependencies
pip install torch transformers datasets peft accelerate bitsandbytes

3️⃣ Run fine-tuning

Open the Colab notebook (notebook.ipynb) and run all cells.
This will:

Load the base Mistral-7B-Instruct model

Preprocess the Legal Q&A dataset

Fine-tune with LoRA adapters

Save the trained model in /content/legal-llm
4️⃣ Inference Example
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/content/legal-llm"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = "What is the difference between a contract and an agreement?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
