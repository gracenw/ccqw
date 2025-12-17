from transformers import AutoTokenizer
from models.modeling_opt_local import OPTForCausalLM

def main():
    model = OPTForCausalLM.from_pretrained("facebook/opt-350m", attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    print(model)

    model_inputs = tokenizer([("What are we having for dinner?")], return_tensors="pt")

    generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=True)
    print(tokenizer.batch_decode(generated_ids)[0])

    # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    # model = OPTForCausalLM.from_pretrained("facebook/opt-13b", attn_implementation="sdpa", quantization_config=bnb_config)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

    # generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
    # print(tokenizer.batch_decode(generated_ids)[0])

    # generator = pipeline('text-generation', model="facebook/opt-2.7b")

    # print(generator("What are we having for dinner?"))

if __name__ == '__main__':
    main()