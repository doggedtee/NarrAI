from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)

llm_haiku = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)

llm_gpt4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)
