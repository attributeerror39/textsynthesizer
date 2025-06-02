from openai import OpenAI

def run_llm(system_prompt, prompt, local=True):

    if local == True:
        
        # Local ollama
        openai_api_base = 'http://localhost:11434/v1'
        model = 'mistral-small:latest'
    
    else:

        # LLM from our server in Mainz
        openai_api_base = "https://kind-mistral.service.kitegg.hs-mainz.de:30443/v1"
        model = "mistralai/Mistral-Small-Instruct-2409"


    # Create a client to connect with the model
    client = OpenAI(
        api_key='EMPTY',
        base_url=openai_api_base,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    # print("Chat response:", response.choices[0].message.content)
    
    # yield response.choices[0].message.content

    output = ''
    for chunk in response:

        print(chunk.choices[0].delta.content, end='')

        output += chunk.choices[0].delta.content
        yield output
        
# For testing, if this python file is executed alone.
run_llm('you are a pirate', 'how are you today?')