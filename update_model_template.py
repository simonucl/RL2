# from transformers import AutoTokenizer, AutoModelForCausalLM

# def update_and_demonstrate_template(model_name: str):
#     """
#     Loads a tokenizer, updates its chat template, and demonstrates the result.

#     Args:
#         model_name: The name of the model on the Hugging Face Hub.
#     """
#     # 1. Load the tokenizer and model from the Hugging Face Hub
#     print(f"Loading tokenizer for '{model_name}'...")
#     # Using trust_remote_code=True may be necessary for some custom models
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
#     # Load the model to be able to push it to the hub
#     print(f"Loading model '{model_name}'...")
#     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

#     print("\nOriginal chat template:")
#     print(tokenizer.chat_template)

#     # 2. Define the new, task-specific chat template as a Jinja string
#     # This template hardcodes the system prompt and the user prompt's structure
#     # as per your request. The `{{ message['content'] }}` part is where the
#     # actual user input (the math problem) will be inserted.
#     new_chat_template = (
#         "{{ bos_token }}"
#         "<|system|>\n"
#         "You are a helpful assistant.\n"
#         "{% for message in messages %}"
#             "{% if message['role'] == 'user' %}"
#                 "<|user|>\n"
#                 "Question: {{ message['content'] }}\n"
#                 "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
#             "{% elif message['role'] == 'assistant' %}"
#                 "<|assistant|>\n"
#                 "{{ message['content'] }}{{ eos_token }}\n"
#             "{% endif %}"
#         "{% endfor %}"
#         "{% if add_generation_prompt %}"
#             "<|assistant|>\n"
#         "{% endif %}"
#     )

#     # 3. Update the tokenizer's chat_template attribute
#     tokenizer.chat_template = new_chat_template

#     print("\n--- TEMPLATE UPDATED ---")
#     print("\nNew chat template:")
#     print(tokenizer.chat_template)

#     # 4. Demonstrate the new template with an example
#     print("\n--- DEMONSTRATION ---")
#     example_messages = [
#         {"role": "user", "content": "What is 2+2?"}
#     ]
    
#     formatted_prompt = tokenizer.apply_chat_template(
#         example_messages, 
#         tokenize=False, 
#         add_generation_prompt=True
#     )

#     print("\nExample prompt generated with the new template:")
#     print(formatted_prompt)

#     # 5. (Optional) Save the updated tokenizer (and model) locally
#     # print("\nSaving updated tokenizer to './olmoe-updated-template'...")
#     # tokenizer.save_pretrained("./olmoe-updated-template")
#     # model.save_pretrained("./olmoe-updated-template")

#     # 6. Push the updated tokenizer and model to the Hugging Face Hub
#     # You must be logged in via `huggingface-cli login` for this to work.
#     print("\nPushing to the Hub...")
#     hub_repo_id = "simonycl/OLMoE-1B-7B-0125-Instruct"
#     tokenizer.push_to_hub(hub_repo_id)
#     model.push_to_hub(hub_repo_id)
#     print(f"Successfully pushed model and tokenizer to '{hub_repo_id}'")


# if __name__ == "__main__":
#     model_id = "allenai/OLMoE-1B-7B-0125-Instruct"
#     update_and_demonstrate_template(model_id)

from transformers import AutoTokenizer, AutoModelForCausalLM

def update_and_demonstrate_template(model_name: str):
    """
    Loads a tokenizer, updates its chat template, and demonstrates the result.

    Args:
        model_name: The name of the model on the Hugging Face Hub.
    """
    # 1. Load the tokenizer and model from the Hugging Face Hub
    print(f"Loading tokenizer for '{model_name}'...")
    # Using trust_remote_code=True may be necessary for some custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load the model to be able to push it to the hub
    print(f"Loading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    print("\nOriginal chat template:")
    print(tokenizer.chat_template)

    # 2. Define the new, task-specific chat template as a Jinja string
    # Using {% raw %} ... {% endraw %} tells Jinja to ignore the {{}} inside the \boxed{} command
    new_chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "A conversation between User and Assistant. The user asks a question, and "
                "the Assistant solves it. The assistant first thinks about the reasoning process in the mind and "
                "then provides the user with the answer. User: You must put your answer inside {% raw %}\\boxed{{}}{% endraw %} "
                "and your final answer will be extracted automatically by the {% raw %}\\boxed{{}}{% endraw %} tag.\n"
                "Question: {{ message['content'] }}\n"
                "Assistant:"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}\n\n"
            "{% endif %}"
        "{% endfor %}"
    )


    # 3. Update the tokenizer's chat_template attribute
    tokenizer.chat_template = new_chat_template

    print("\n--- TEMPLATE UPDATED ---")
    print("\nNew chat template:")
    print(tokenizer.chat_template)

    # 4. Demonstrate the new template with an example
    print("\n--- DEMONSTRATION ---")
    example_messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        example_messages, 
        tokenize=False, 
        add_generation_prompt=False # No generation prompt needed for this format
    )

    print("\nExample prompt generated with the new template:")
    print(formatted_prompt)

    # 5. (Optional) Save the updated tokenizer (and model) locally
    # print("\nSaving updated tokenizer to './olmoe-updated-template'...")
    # tokenizer.save_pretrained("./olmoe-updated-template")
    # model.save_pretrained("./olmoe-updated-template")

    # 6. Push the updated tokenizer and model to the Hugging Face Hub
    # You must be logged in via `huggingface-cli login` for this to work.
    print("\nPushing to the Hub...")
    hub_repo_id = "simonycl/OLMoE-1B-7B-0125"
    tokenizer.push_to_hub(hub_repo_id)
    model.push_to_hub(hub_repo_id)
    print(f"Successfully pushed model and tokenizer to '{hub_repo_id}'")


if __name__ == "__main__":
    model_id = "allenai/OLMoE-1B-7B-0125"
    update_and_demonstrate_template(model_id)