from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langsmith import traceable
from dotenv import load_dotenv
from langchain.messages import SystemMessage,HumanMessage,ToolMessage

load_dotenv()

MAX_ITERATIONS =10

@tool
def get_product_price(product : str) -> float:
    """ method to fetch the product price"""
    print(f'fetching the price of the product :{product}')
    product_details= {"laptop":13456.7,"mouse":234.5,"printer":678.9}
    return product_details.get(product,0)

@tool
def get_final_price_of_the_product_after_discount(price: float,member_type:str)-> float:
    """ method to get the final price of the product after applying discount"""
    print(f'fetching the final price of the product after applying discount : {member_type}')
    member_ship= {"silver":10,"bronze":15,"gold":20}
    discount =member_ship.get(member_type,0)
    final_price= round(price * (1-discount/100),2)
    return final_price

@traceable(name="LANG CHAIN AGENT GROUP")
def run_agent(question:str):
    print(f'executing get_tracing')
    tools = [get_product_price,get_final_price_of_the_product_after_discount]
    tools_dict = {t.name: t for t in tools}
    llm=init_chat_model( "google_genai:gemini-3.6-flash")
    llm_with_tools=llm.bind_tools(tools)

    print(f'question : {question}')

    messages = [
        SystemMessage( content =
                       (
                           "You are a helpful shopping assistant. "
                           "You have access to a product catalog tool "
                           "and a get_final_price_of_the_product_after_discount tool.\n\n"
                           "STRICT RULES — you must follow these exactly:\n"
                           "1. NEVER guess or assume any product price. "
                           "You MUST call get_product_price first to get the real price.\n"
                           "2. Only call get_final_price_of_the_product_after_discount AFTER you have received "
                           "a price from get_product_price. Pass the exact price "
                           "returned by get_product_price — do NOT pass a made-up number.\n"
                           "3. NEVER calculate discounts yourself using math. "
                           "Always use the get_final_price_of_the_product_after_discount tool.\n"
                           "4. If the user does not specify a discount tier, "
                           "ask them which tier to use — do NOT assume one."
                           "5. Always consider price is in INR"
                        )
                        ),
        HumanMessage(content =question)
                ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ =='__main__':
    message ="What is the price of a laptop after applying a gold discount?"
    run_agent(message)