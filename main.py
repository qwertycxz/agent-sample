from collections.abc import Iterable
from contextlib import asynccontextmanager
from logging import getLogger

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.mcp import HttpStatelessClient
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import stream_printing_messages
from agentscope.tool import ToolResponse, Toolkit
from agentscope_runtime.engine.app import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest, AgentResponse
from agentscope_runtime.engine.tracing.base import EventContext
from fastapi import FastAPI
from tiktoken import get_encoding

from cst import CSTKnowledgeBase

LOGGER = getLogger('智能体')

def calculate(operator: str, operand1: float, operand2: float):
	'''
	四则运算计算器

	Args:
		operator (str): 运算符，可为加（+）、减（-）、乘（*）、除（/）
		operand1 (float): 操作数1
		operand2 (float): 操作数2
	'''
	result: float
	match operator:
		case '+':
			result = operand1 + operand2
		case '-':
			result = operand1 - operand2
		case '*':
			result = operand1 * operand2
		case '/':
			result = operand1 / operand2
		case _:
			raise ValueError(f'运算符不支持: {operator}')
	return ToolResponse([
		TextBlock(text = f'{result}', type = 'text')
	])

KNOWLEDGE = 'AI电子书'
knowledge = CSTKnowledgeBase('Bearer ', '', KNOWLEDGE)

mcp = HttpStatelessClient('MCP', 'sse', '', {
	'Authorization': 'Bearer '
})

toolkit = Toolkit()

@asynccontextmanager
async def lifespan(app: FastAPI):
	toolkit.register_tool_function(calculate)
	toolkit.register_tool_function(knowledge.retrieve_knowledge, func_description = KNOWLEDGE)
	await toolkit.register_mcp_client(mcp)
	yield

app = AgentApp(lifespan = lifespan)
encoding = get_encoding('o200k_base')
formatter = OpenAIChatFormatter()
model = OpenAIChatModel('qwen3.5', '', client_kwargs = {
	'base_url': 'https://uni-api.cstcloud.cn/v1',
})

@app.query()
async def query(self, msgs: Iterable[Msg], request: AgentRequest, response: AgentResponse, trace_event: EventContext):
	agent = ReActAgent('生态环境智能体', '你是生态环境智能体', model, formatter, toolkit)
	agent.set_console_output_enabled(False)
	try:
		async for messages in stream_printing_messages([
			agent,
		], agent(msgs)):
			yield messages
	except:
		await agent.interrupt()
	LOGGER.warning(f'估计消耗Token：{sum(len(encoding.encode(str(memory.content))) for memory in await agent.memory.get_memory())}')

if __name__ == '__main__':
	app.run()
