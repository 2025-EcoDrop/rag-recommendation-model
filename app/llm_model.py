import os

IS_TEST = os.getenv("TEST_ENV") == "true"

if not IS_TEST:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "saltlux/Ko-Llama3-Luxia-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

async def generate_answer(query: str, docs: list[list[str]]) -> str:
    if not IS_TEST:
        context = "\n".join(
            f"{i + 1}. " + " ".join(doc)
            for i, doc in enumerate(docs)
        )
        prompt = f"질문: {query}\n참고 자료: {context} + 응답에는 \'참고용, 전문가 상담 필요\' 문구 포함\n답변: "

        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            temperature=0.7,
            no_repeat_ngram_size=6,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, '')

    return "test"
