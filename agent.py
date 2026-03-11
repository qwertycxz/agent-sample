from collections.abc import Iterable

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

HttpStatelessClient

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

toolkit = Toolkit()

toolkit.register_tool_function(calculate)

app = AgentApp()

@app.query()
async def query(self, msgs: Iterable[Msg], request: AgentRequest, response: AgentResponse, trace_event: EventContext):
	agent = ReActAgent('生态环境智能体', '你是生态环境智能体', OpenAIChatModel('qwen3.5', '', client_kwargs = {
		'base_url': 'https://uni-api.cstcloud.cn/v1',
	}), OpenAIChatFormatter(), toolkit)
	async for messages in stream_printing_messages([
		agent,
	], agent(msgs)):
		yield messages

if __name__ == '__main__':
	app.run()
