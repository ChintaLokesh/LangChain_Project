
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv


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


def create_tool_for_product(product_name:str,discount_variant:str):
    llm=ChatGoogleGenerativeAI(model ='gemini-3.6-flash')
    tools =[get_product_price,get_discount_price_of_the_product]
    message=HumanMessage(content =f" get the price of {product_name} with discount {discount_variant}")
    system_prompt = (
        "you are an AI assistant to find the price of the product and then get the final price of the product after discount"
        "1 . dont assume price of the product"
        "2 . if product is not available then return product not available"
        "3. dont do the calculations by yourself follow the instructions carefully"
    )
    agent=create_agent(llm,tools,system_prompt=system_prompt)
    response=agent.invoke({"messages":[message]})
    print(f'response:{response['messages'][-1].content}')


if __name__ == '__main__':
    create_tool_for_product('apple macbook','Gold')
