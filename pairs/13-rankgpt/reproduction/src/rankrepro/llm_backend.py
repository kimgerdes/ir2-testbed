"""Hugging Face causal-LM backend for listwise permutation generation (RankZephyr and friends)."""
import time

import torch


class HFChatLLM:
    """Greedy chat generation with optional bitsandbytes 4/8-bit loading.

    The tokenizer is loaded immediately (cheap, CPU); the model weights are loaded on the first generate() call, so
    a run whose windows are all cached (e.g. the course's reproduce.ipynb replaying our cache) never touches the GPU.
    """

    def __init__(self, model_name, quant="4bit", dtype="float16", device_map="auto", max_memory=None, double_quant=False,
                 lazy=True):
        """dtype is the compute dtype. On an 8 GB GPU, float16 without double quantisation peaked at 4.7 GB for
        RankZephyr-7B, while bfloat16 + double quant peaked above 8 GB and spilled into system memory (20x slower)."""
        from transformers import AutoTokenizer

        self.name = model_name
        self.quant = quant
        self.dtype = dtype
        self.device_map = device_map
        self.max_memory = max_memory
        self.double_quant = double_quant
        self.model = None
        self.device = None
        self.load_seconds = 0.0
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if not lazy:
            self.load_model()

    def load_model(self):
        if self.model is not None:
            return self.model
        import transformers
        from transformers import AutoModelForCausalLM

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
        kwargs = {"device_map": self.device_map}
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if self.quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if self.quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                   bnb_4bit_compute_dtype=torch_dtype,
                                                                   bnb_4bit_use_double_quant=self.double_quant)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kwargs[dtype_key] = torch_dtype
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **kwargs).eval()
        self.load_seconds = time.time() - t0
        self.device = next(self.model.parameters()).device
        print(f"model {self.name} ({self.quant}) loaded in {self.load_seconds:.0f}s on {self.device}", flush=True)
        return self.model

    @property
    def loaded(self):
        return self.model is not None

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=160, prefill_chunk_size=512):
        """Greedy generation. prefill_chunk_size bounds the prefill activations: on an 8 GB GPU a 4k-token prompt
        peaked at 9.4 GB without chunking (the driver spilled to system memory, 10x slower) and 4.9 GB with it."""
        model = self.load_model()
        prompt = self.render(messages)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        n_prompt = int(enc.input_ids.shape[1])
        t0 = time.time()
        extra = {"prefill_chunk_size": prefill_chunk_size} if prefill_chunk_size else {}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=self.tok.pad_token_id, **extra)
        gen = out[0, n_prompt:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        return {"text": text, "prompt_tokens": n_prompt, "gen_tokens": int(gen.shape[0]), "seconds": time.time() - t0,
                "prompt": prompt}

    def free(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
