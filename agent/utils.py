import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

class TokenTracker(BaseCallbackHandler):
    def __init__(self):
        self.reset()

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_calls = 0
        self.estimated_cost_usd = 0.0

    def on_llm_end(self, response, **kwargs):
        """Collect token usage from the response."""
        for generations in response.generations:
            for generation in generations:
                if hasattr(generation, 'message'):
                    message = generation.message
                    if hasattr(message, 'usage_metadata') and message.usage_metadata:
                        usage = message.usage_metadata
                        self.input_tokens += usage.get('input_tokens', 0)
                        self.output_tokens += usage.get('output_tokens', 0)
                        self.total_calls += 1
                        # Gemini 2.5 Flash Lite approximate costs
                        self.estimated_cost_usd = (self.input_tokens * 0.10 / 1e6) + (self.output_tokens * 0.40 / 1e6)

    def summary(self):
        return f"Tokens: In={self.input_tokens}, Out={self.output_tokens} | Calls: {self.total_calls}"

token_tracker = TokenTracker()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=6,
    timeout=60,
)

def _llm(system: str, user: str, agent: str = "Research Agent") -> str:
    """Invokes the LLM with retry logic for 429 errors."""
    for attempt in range(3):
        try:
            result = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ], config={"callbacks": [token_tracker]})
            return result.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait_time = (attempt + 1) * 5
                time.sleep(wait_time)
            else:
                raise e
    return ""
