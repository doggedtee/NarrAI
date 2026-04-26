from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
)
