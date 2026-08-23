from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage,SystemMessage,ToolMessage
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()



product_list= [{"sony mobile":30000,"discount_variant":"Silver"},{"apple macbook":45000,'discount_variant':"Bronze"},{'oneplus tab':40000,'discount_variant':"Gold"}]

@tool
def get_product_price(product_name:str):
    """ get the price of the product """

    selected_product_price = next((product[product_name] for product in product_list if product_name in product),"Not Found")
    print(f'invoking the method get_product_price with product name {product_name}')
    print(f'selected product price :{selected_product_price}')
    return int(selected_product_price)


@tool
def get_discount_price_of_the_product(product_price:int,discount_variant:str):
    """ get the price of the product after applying the discount variant """
    print(f'invoking the method get_discount_price_of_the_product with product price {product_price} having discount variant :{discount_variant}')
    if discount_variant == "Bronze":
        product_price = product_price - (product_price * (2.5/100))
        print(f'product final price along with the discount :{product_price}')

    elif discount_variant == "Silver":
        product_price = product_price - (product_price * (5/100))
        print(f'product final price along with the discount :{product_price}')

    elif discount_variant == "Gold":
        product_price = product_price - (product_price * (10/100))
        print(f'product final price along with the discount :{product_price}')
    return product_price


@traceable(name="LANG CHAIN AGENT GROUP")
def create_tool_calling(product_name : str,discount_variant:str ):
    MAX_ITERATIONS = 10
    messages = [

        SystemMessage(content ="you are an AI assistant to find the price of the product and then get the final price of the product after discount"
        "1 . dont assume price of the product"
        "2 . if product is not available then return product not available"
        "3. dont do the calculations by yourself follow the instructions carefully"),

        HumanMessage(content =f"give the final price of {product_name} having {discount_variant}")



    ]
    tools =[get_product_price,get_discount_price_of_the_product]
    tools_dict={t.name:t for t in tools}
    # tools_dict = {t.name: t for t in tools}
    llm=init_chat_model("google_genai:gemini-3.6-flash")
    llm_with_tools=llm.bind_tools(tools)

    for iteration in range(MAX_ITERATIONS):

        response=llm_with_tools.invoke(messages)
        print(f'response:{response}')

        tool_calls=response.tool_calls

        if not tool_calls:
            print(f'final response:{response.content}')
            return response.content[-1]['text']
        if len(tool_calls) > 0:
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

            messages.append(response)
            messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )

    # print("ERROR: Max iterations reached without a final answer")
    # return None



if __name__ == '__main__':
    create_tool_calling("sony mobile","Silver")