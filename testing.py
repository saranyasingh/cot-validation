from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input='''
I will give you a murder story. Provide a logical argument for who is the most likely murderer. Explain every step of your logical reasoning in the format of a logical syllogism where each premise follows from the previous one, with enough context where I can understand your logic even without the full story. End your answer with The murderer is: X where X is your final answer.

At 9 pm, Alice was in the kitchen. Bob was in the garden holding a knife.
Clara saw Bob leave the study shortly before the murder.
The victim, Mr. Black, was found dead in the study.
'''
)

print(response.output_text)
