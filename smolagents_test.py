from smolagents import CodeAgent, DuckDuckGoSearchTool, load_tool, tool, Tool
import datetime
import json
import requests
import pytz
import yaml
import openai
from openai import OpenAI
import os

client = OpenAI(base_url=os.getenv("OPENAI_API_BASE", "http://YOUR_LLAMA_CPP_BASE_URL:8080"))

inventory = {
    "apple": 3,
    "orange": 1,
}

@tool
def inventory_updater(objectName: str, quantityModifier: int) -> str:
    """A tool that will update the user's inventory if a story event occurs. Returns
    whether or not the inventory update was successful. object names will always be in the 
    singular.
    Args:
        objectName: the name of the object being updated (example: 'apple')
        quantityModifier: How much to increase or decrease the number held (positive if adding, negative if removing)
    """
    if objectName in inventory:
        newQuantity = inventory[objectName] + quantityModifier
    else:
        newQuantity = quantityModifier
    if newQuantity < 0:
        return "error! this would leave you with {newQuantity} of {objectName} in inventory. It cannot be done!"
    inventory[objectName] = newQuantity
    return f"Success - you now have {newQuantity} of {objectName}"

@tool
def fetch_inventory() -> str:
    """A tool that will fetch the user's current inventory in dictionary (JSON) format"""
    return json.dumps(inventory)

# === Define tools ===
class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Use this tool to return the final answer to the user after reasoning is complete."
    inputs = {
        "answer": {
            "type": "string",
            "description": "The final answer to return to the user.",
        }
    }
    output_type = "string"

    def forward(self, answer):
        return f"Final Answer: {answer}"


@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def multiply_two_numbers(num1: int, num2: int) -> int:
    """A tool that takes in two integers and multiplies them, returning the product.
    Args:
        num1: the first number to multiply
        num2: the second number to multiply
    """
    return num1 * num2


# === Setup essential tools ===
final_answer = FinalAnswerTool()
duckduckgo_tool = DuckDuckGoSearchTool()


# === Define local OpenAI-compatible model wrapper ===
class LocalOpenAICompatibleModel:
    def __init__(self, model_id: str, max_tokens: int = 2048, temperature: float = 0.7):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, messages, **kwargs):
        return self.generate(messages, **kwargs)

    def generate(self, messages, **kwargs):
        kwargs.pop("stop_sequences", None)

        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        return response.choices[0].message


model = LocalOpenAICompatibleModel(
    model_id="mikes-model",
    max_tokens=2096,
    temperature=0.5,
)

# Load the HF prompt templates ===
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

# === 6. Create the agent ===

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        duckduckgo_tool,
        get_current_time_in_timezone,
        my_custom_tool,
        multiply_two_numbers,
        inventory_updater,
        fetch_inventory,
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="local_llm_agent",
    description="An agent that uses a local OpenAI-compatible model.",
    prompt_templates=prompt_templates,
)

if __name__ == "__main__":
    print("🧠 Local Agent CLI is running. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            # Run the agent's reasoning loop
            response = agent.run(user_input)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
