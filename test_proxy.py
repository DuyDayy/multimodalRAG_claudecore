import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

print("BASE URL:", os.getenv("OPENAI_BASE_URL"))
print("API KEY:", str(os.getenv("OPENAI_API_KEY"))[:10] + "...")

try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    res = llm.invoke([HumanMessage(content="Hello")])
    print("SUCCESS:", res.content)
except Exception as e:
    print("ERROR:", type(e).__name__, "-", e)
