import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()


CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


class ChatModel:
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cpu"):

        ACCESS_TOKEN = os.getenv(
            "ACCESS_TOKEN"
        )  # reads .env file with ACCESS_TOKEN=<your hugging face access token>

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # device_map="auto",
            # quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):

        if context == None or context == "":
            prompt = f"""
            Esti un reprezentativ BCR Romania care ofera informatii utile clientilor. Intrebare client: {question}"""
        else:
            prompt = f"""
Esti un reprezentativ BCR Romania care ofera informatii utile clientilor.
Iti dau intrebarea clientului si tu ii vei raspunde cat mai bine bazat pe informatiile din documentele date.
Trimit la client bazat pe informatia din documente si pe baza la urmatoarele reguli:

1/ Nu ai voie sa iei date din surse externe.
2/ Raspunsul trebuie sa fie un rezumat precis al datelor usor de inteles.
3/ Daca datele din documente nu sunt relevante, spune ca nu ai informatii.
4/ Precizeaza si din ce articol ai luat informatia.
5/ Structureaza raspunsul in bullet points.

Mesajul de la client este:
{context}

Aici sunt date din documente:
{question}

Te rog sa scrii raspunsul cel mai bun pe care sa il trimit clientului:
"""

        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(formatted_prompt)
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt) :]  # remove input prompt from reponse
        response = response.replace("<eos>", "")  # remove eos token

        return response
