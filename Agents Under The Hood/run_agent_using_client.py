

from google.genai import Client
from google.genai.types import GenerateContentConfig
import re
from dotenv import load_dotenv
load_dotenv()
client = Client()

def calculator(expression : str) -> str:
    """ method used to calculate the given expression
    args : expression (str) : Mathematical expression for instance "23 * 4 " or "( 2 * 3 ) / 10 "
    returns :
        str : Numeric result or an error message starting with Error message
    """
    try:
        return str(eval(expression,{"__builtins__":{}},{}))
    except Exception as e:
        return f" Error message : {e}"


def get_climate(city_name : str) -> str:
    """ method used to get the climate of the given city
    args : city_name (str) : city name
    returns :
    str : climate of the given city
    """
    data = {
         "chennai": "32°C, humid, partly cloudy",
        "bangalore": "24°C, pleasant, light rain",
        "delhi": "28°C, hazy",
        "mumbai": "30°C, humid"


    }
    return data.get(city_name.lower(),f" no weather for city name : {city_name}")

def word_count(text : str) -> str:
    """ method used to count the number of words in the given text"""
    return str(len(text.split()))

TOOLS = {
   "calculator": calculator,
   "get_climate": get_climate,
    "word_count": word_count
}


TOOL_DESCRIPTIONS = """
- calculator(expression: str) -> str
    Evaluate a math expression. Example: calculator("23 * 47")
- get_weather(city: str) -> str
    Return current weather for a city. Example: get_weather("Chennai")
- word_count(text: str) -> str
    Count words in text. Example: word_count("hello world")
"""

SYSTEM_PROMPT = f"""You are a ReAct agent that solves problems step by step.

Tools available:
{TOOL_DESCRIPTIONS}

Format (follow exactly):

Thought: <reasoning>
Action: <tool_name>
Action Input: <input string>

After Action, STOP. The system replies with:

Observation: <result>

Continue with another Thought/Action, or finish:

Thought: I now know the final answer.
Final Answer: <answer>

Rules:
- One Thought + Action per turn, then wait.
- Action must be one of: {list(TOOLS.keys())}
- Action Input is a plain string (no quotes).
- Never invent Observations.
"""




def parse_response(text: str):
    """Parse LLM output into ('final', answer) | ('action', name, input) | ('error', msg)."""
    if m := re.search(r"Final Answer:\s*(.+)", text, re.DOTALL):
        return "final", m.group(1).strip()

    a = re.search(r"Action:\s*(.+)", text)
    i = re.search(r"Action Input:\s*(.+)", text)
    if a and i:
        return "action", a.group(1).strip(), i.group(1).strip().strip('"').strip("'")

    return "error", "Could not parse Action or Final Answer."



def run_agent(question:str,max_iterations=10):
    chat=client.chats.create(model = "gemini-3.6-flash",
                        config=GenerateContentConfig(system_instruction=SYSTEM_PROMPT,temperature=0,stop_sequences=['Observation:']))
    agent_response=chat.send_message(question)
    print(f'user input : {question}')
    for iteration in range(max_iterations):
        response=agent_response.text
        print(f'iteration : {iteration+1} agent response : {response}')
        parsed_response=parse_response(response)
        if parsed_response[0] =="final":
            final_answer= parsed_response[1]
            break
        if parsed_response[0] == "error":
            final_answer= parsed_response[1]
            break

        _,name,args = parsed_response
        if name in TOOLS:
            try:
                observation=TOOLS[name](args)
                agent_response = chat.send_message(f'Observation: {observation}')
            except Exception as e:
                print(f'exception occurred : {e}')
        else :
            print(f'No tool available with name {name} available tools are : {TOOLS}')

    else:
        print(f'iteration exceeded and the required answer is not obtained from agent')
        final_answer =f'iteration exceeded and the required answer is not obtained from agent'

    return final_answer

run_agent("what is the result of 12 * 6 / 2")















