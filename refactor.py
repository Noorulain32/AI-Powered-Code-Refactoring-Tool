import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class CodeRefactor:
    def __init__(self, model_path="model/codeT5_finetuned"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

    def refactor(self, input_code: str) -> str:
        # Prepare prompt for CodeT5
        prompt = f"refactor Python: {input_code.strip()}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                padding="max_length", max_length=512).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            no_repeat_ngram_size=2
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_output(result)

    def _clean_output(self, output: str) -> str:
        return output.replace('\\n', '\n').replace('\\t', '\t').strip()

# Export auto_format here for reusability
import autopep8

def auto_format(code: str) -> str:
    try:
        return autopep8.fix_code(code)
    except Exception as e:
        return f"❌ Error during auto-formatting: {e}"
