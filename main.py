#!/usr/bin/env python
from collections.abc import Iterable
from contextlib import asynccontextmanager
from logging import getLogger

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import stream_printing_messages
from agentscope.tool import Toolkit
from agentscope_runtime.engine.app import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest, AgentResponse
from agentscope_runtime.engine.tracing.base import EventContext
from fastapi import FastAPI
from tiktoken import get_encoding

from cst import calculate

LOGGER = getLogger('智能体')

toolkit = Toolkit()

@asynccontextmanager
async def lifespan(app: FastAPI):
	'''
	在服务启动前（yield前）和服务结束后（yield后）执行一些代码；建议在这里注册工具调用

	Args:
		app: FastAPI应用实例
	'''
	# 一般本地工具
	toolkit.register_tool_function(calculate)
	LOGGER.info(f'已注册工具：{calculate}')
	yield

# 这里执行一些初始化工作
app = AgentApp(lifespan = lifespan)
encoding = get_encoding('o200k_base')
formatter = OpenAIChatFormatter()
model = OpenAIChatModel('qwen3.5', '', client_kwargs = {
	'base_url': 'https://uni-api.cstcloud.cn/v1',
})

@app.query()
async def query(self: AgentApp, msgs: Iterable[Msg], request: AgentRequest, response: AgentResponse, trace_event: EventContext):
	'''
	智能体处理函数，接收用户输入的消息，返回智能体的回复消息

	Args:
		msgs: 用户输入的消息列表
		request: AgentRequest对象，包含请求的相关信息
		response: AgentResponse对象，用于设置回复消息和其他响应信息
		trace_event: EventContext对象，用于记录智能体处理过程中的事件和日志
	'''
	# 每次都创建一个新 Agent，保证每次请求都是独立的；如果需要在请求之间共享一些状态，可以考虑使用 Agent.memory 或者其他持久化存储，见 https://doc.agentscope.io/zh_CN/tutorial/task_memory.html
	agent = ReActAgent('生态环境智能体', '你是生态环境智能体', model, formatter, toolkit)
	# 如果不需要输出控制台日志，可以关闭；如果需要调试，可以打开
	agent.set_console_output_enabled(False)
	try:
		# 大模型的流式输出
		async for messages in stream_printing_messages([
			agent,
		], agent(msgs)):
			yield messages
	except:
		# 如果用户手动断开连接，那我们就中断智能体
		await agent.interrupt()
	LOGGER.warning(f'估计消耗Token：{sum(len(encoding.encode(str(memory.content))) for memory in await agent.memory.get_memory())}')

# 可以在本地执行 `python main.py` 来启动服务，建议在发布前先在本地进行测试
if __name__ == '__main__':
	app.run()
