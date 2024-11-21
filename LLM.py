import os, requests
# from langchain.llms import HuggingFaceHub
from dotenv import find_dotenv, load_dotenv
# from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate


def get_local_response(prompt: str, model:str="mistral-nemo:latest"):
    """This function takes inputs prompt and a model. 

        A prompt is your query to the LLM and takes input a string.
        Model is the open-source ollama model you want to use, it takes input a string.

        If a model is not provided, llama3.1:70b is taken as a default model.
    """
    try:
        response = requests.post(
            "http://192.168.25.131:8008/generate",
            json={"prompt": prompt, "model": model}
        ).json()
        return response['response']
    except Exception as e:
        print(f"Encountered unexpected error: {e}")
        return None


# load_dotenv(find_dotenv())
# huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7b-Instruct-v0.1", model_kwargs={'temperature':0.4})
# import google.generativeai as palm
# api_key=os.environ["GOOGLE_API_KEY"]
# palm.configure(api_key=api_key)

# models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
# model = models[0].name
# print(model)


def askLLM(amount,sndr_ini_blc,sndr_fin_blc,rec_ini_blc,rec_fin_blc,type,ans):
    inputs=f"""
        A transaction of type {type} has happened, an amount of ${amount} is debited from the sender\'s account whose balance is ${sndr_ini_blc}. The new balance in the sender\'s account is ${sndr_fin_blc}. The reciever who was initially having ${rec_ini_blc} in his account is now having ${rec_fin_blc}
    """
    Prompt=f"""You are a financial-fraud assistant. You have to only predict if the given transaction is fraud or not. Answer only if you can decide the type of transaction. Do not makeup any information if not provided.
        {inputs}
    """
    
    
    from langchain_core.prompts import PromptTemplate
    
    
    Prompt2=f"""
    You are a Finance officer, detecting whether a transaction is fraud or not. I have a machine learning model that predicts the likelihood of fraud for each transaction.\n 
    
    
    Here are the details of a transaction:\n
    Amount: ${amount}\n
    Sender bank balance before transaction: ${sndr_ini_blc}\n
    Sender bank balance after transaction: ${sndr_fin_blc}\n
    Reciever bank balance before transaction: ${rec_ini_blc}\n
    Reciever bank balance after transaction: ${rec_fin_blc}\n
    Transaction type: {type}\n
    
    Model Output:{ans}\n  Please justify why this prediction is such way based on the details provided. Make the reason be unbiased.
    """
    # result=llm(Prompt2)
    
    
    
    Prompt3="""
    I am an AI analyzing transactions for fraud. I have a machine learning model that predicts the likelihood of fraud for each transaction. Here are the details of a transaction:\n
    
    Transaction Details:
    Amount: ${amount}\n
    Sender bank balance before transaction: ${sndr_ini_blc}\n
    Sender bank balance after transaction: ${sndr_fin_blc}\n
    reciever bank balance before transaction: ${rec_ini_blc}\n
    reciever bank balance after transaction: ${rec_fin_blc}\n
    Transaction type: {type}\n
    
    Model Output:{ans}  Please analyze this transaction and share your thoughts on the fraud risk. Consider the model's prediction, but don't rely solely on it. Use your understanding of real-world scenarios and potential fraud patterns to provide insights.
    """
    # prompt = PromptTemplate(input_variables=["amount", "sndr_ini_blc","sndr_fin_blc","rec_ini_blc","rec_fin_blc","type","ans"], template=Prompt3)
    # # result=llm(Prompt)
    # chain = LLMChain(llm=llm,prompt=prompt)
    # result=chain.run({"amount":amount,"sndr_ini_blc":sndr_ini_blc,"sndr_fin_blc":sndr_fin_blc,"rec_ini_blc":rec_ini_blc,"rec_fin_blc":rec_fin_blc,"type":type,"ans":ans})
    # chain = LLMChain(llm, prompt=Prompt)
    

    result=get_local_response(Prompt2)

    
    # text = palm.generate_text(
    # prompt=Prompt2,
    # model=model,
    # temperature=0.3,
    # max_output_tokens=64,
    # top_p=0.9,
    # top_k=40,
    # stop_sequences=['\n']
    # )
    # result=text.result    
    return result


def askLLM2(amount,sndr_ini_blc,sndr_fin_blc,type,ans):
    # Prompt="""Input:
    #     Please analyze the following transaction details and determine whether the transaction is fraudulent or not. Do not generate information beyond what is provided in the transaction details.

    #     Transaction Details:

    #     Transaction Amount: ${amount}
    #     Sender Initial Balance: ${sndr_ini_blc}
    #     Sender Final Balance: ${sndr_fin_blc}
    #     Transaction Type: {type}

    #     Output:

    #     Is the transaction fraudulent? (Yes/No)
    #     Provide a brief explanation or reasoning based on the given details.
    # """
    Prompt2=f"""Input:
        Please analyze the following transaction details and determine whether the transaction is fraudulent or not. Do not generate information beyond what is provided in the transaction details. Here is the prediction from my working ML algorithm: {ans}, Consider the model's prediction, but don't rely solely on it.\n

        Transaction Details:
        Transaction Amount: ${amount}\n
        Sender Initial Balance: ${sndr_ini_blc}\n
        Sender Final Balance: ${sndr_fin_blc}\n
        Transaction Type: {type}\n

        Answer the following:
        Is the transaction fraudulent? (Fraud/Not Fraud)\n
        Provide a brief explanation or reasoning for your answer.
    """
    Prompt3=f"""
    I am an AI analyzing transactions for fraud. I have a machine learning model that predicts the likelihood of fraud for each transaction. Here are the details of a transaction:
    
    Transaction Details:
    Amount: ${amount}
    Sender bank balance before transaction: ${sndr_ini_blc}
    Sender bank balance after transaction: ${sndr_fin_blc}
    Transaction type: {type}
    
    Output:
    Model Output:{ans}  Please analyze this transaction and share your thoughts on the fraud risk. Consider the model's prediction, but don't rely solely on it. 
    """

    # Query = PromptTemplate.from_template(template=Prompt)#,input_variables=["amount", "sndr_ini_blc","sndr_fin_blc","type"],)
    # Prompt=Query.format(amount=amount, sndr_ini_blc=sndr_ini_blc,sndr_fin_blc=sndr_fin_blc,type=type)
    # # chain = LLMChain(llm=llm,prompt=Prompt)
    # # result=chain.run(Prompt)
    

    result=get_local_response(Prompt2)

    # # result=llm.invoke(Prompt2)
    # text = palm.generate_text(
    # prompt=Prompt2,
    # model=model,
    # temperature=0.3,
    # max_output_tokens=64,
    # top_p=0.9,
    # top_k=40,
    # stop_sequences=['\n']
    # )
    # result=text.result
    return result