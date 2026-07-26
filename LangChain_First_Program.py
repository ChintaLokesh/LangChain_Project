
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import *

import os

from dotenv import load_dotenv

load_dotenv()

def first_model():
    summary_template = " Given the information {information} convert to {language} language"
    summary_prompt_template=PromptTemplate.from_template(summary_template)
    llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature =0)
    # llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    chain = summary_prompt_template | llm
    result=chain.invoke({"information":"I am a boy","language":"kannada"})
    print(f'result: {result.content[0]['text']}')


if __name__=="__main__":
    first_model()

