from agentscope_runtime.engine.app import AgentApp
from agentscope_runtime.engine.helpers.agent_api_builder import ResponseBuilder
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastmcp import Client, FastMCP
from json import dumps, loads
from logging import getLogger
from mcp import Tool
from mcp.types import AudioContent, ImageContent, TextContent
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageFunctionToolCallParam, ChatCompletionMessageParam, ChatCompletionToolUnionParam
from pydantic import BaseModel, ConfigDict, Field
from tiktoken import get_encoding
from typing import Annotated, Any

LOGGER = getLogger('智能体')
MODEL = 'qwen3.5'

def getTool(tool: Tool) -> ChatCompletionToolUnionParam:
	description = ''
	if tool.description:
		description = tool.description
	return {
		'function': {
			'description': description,
			'name': tool.name,
			'parameters': tool.inputSchema,
		},
		'type': 'function',
	}

mcp = FastMCP()

@mcp.tool(description = 'A simple addition tool example for calculating the sum of two integers', name = 'add_Tool')
def addNumbers(a: Annotated[int, Field(description='add a')], b: Annotated[int, Field(description='add b')]):
	return a + b

mcp_asgi_app = mcp.http_app('/')

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
	'''Application lifecycle management - integrate MCP application's lifespan'''
	async with mcp_asgi_app.router.lifespan_context(fastapi_app):
		yield

app = AgentApp(lifespan = lifespan)

app.mount('/mcp', mcp_asgi_app)

@app.get('/')
def readRoot():
	return 'OK'

@app.get('/health')
def healthCheck():
	return 'OK'

class ContentItem(BaseModel):
	data: dict[str, Any] | None = None
	model_config = ConfigDict(extra = 'allow')
	status: str | None = None
	text: str | None = None
	type: str

class MessageItem(BaseModel):
	content: list[ContentItem] | None = None
	model_config = ConfigDict(extra = 'allow')
	role: str
	type: str | None = None

def getMessage(msg: MessageItem) -> ChatCompletionMessageParam | None:
	if not msg.content:
		return
	content_text = ''.join(content_item.text for content_item in msg.content if content_item.type == 'text' and content_item.text)
	if not content_text:
		return
	if msg.role == 'user':
		return {
			'content': content_text,
			'role': 'user',
		}
	if msg.role == 'assistant' and msg.type == 'message':
		return {
			'content': content_text,
			'role': 'assistant',
		}

class ChatRequest(BaseModel):
	input: list[MessageItem]
	session_id: str
	stream: bool | None = True

class ToolCall(ChatCompletionMessageFunctionToolCallParam):
	index: int

async def generateResponse(request_data: ChatRequest):
	'''Generate streaming response - conforms to Bailian Response/Message/Content architecture'''
	client = AsyncOpenAI(api_key = '', base_url = 'https://uni-api.cstcloud.cn/v1')
	messages: list[ChatCompletionMessageParam] = [message for msg in request_data.input if (message := getMessage(msg))]
	response_builder = ResponseBuilder(request_data.session_id, f'resp_{request_data.session_id}')
	yield f'data: {response_builder.created().model_dump_json()}\n\n'
	yield f'data: {response_builder.in_progress().model_dump_json()}\n\n'
	try:
		response: AsyncStream[ChatCompletionChunk]
		try:
			async with Client('http://127.0.0.1:8080/mcp/') as c:
				response = await client.chat.completions.create(messages = messages, model = MODEL, stream = True, tools = map(getTool, await c.list_tools()))
		except Exception as e:
			response = await client.chat.completions.create(messages = messages, model = MODEL, stream = True)
		llm_content = ''
		tool_calls: list[ToolCall] = []
		current_tool_call: ToolCall | None = None
		async for chunk in response:
			if not chunk.choices:
				continue
			if chunk.choices[0].delta.content:
				llm_content += chunk.choices[0].delta.content
			if chunk.choices[0].delta.tool_calls:
				for tool_call_chunk in chunk.choices[0].delta.tool_calls:
					if current_tool_call and current_tool_call['index'] != tool_call_chunk.index:
						tool_calls.append(current_tool_call)
						current_tool_call = None
					if not tool_call_chunk.function:
						continue
					if not current_tool_call:
						current_tool_call = {
							'function': {
								'name': tool_call_chunk.function.name or '',
								'arguments': tool_call_chunk.function.arguments or '',
							},
							'id': tool_call_chunk.id or '',
							'index': tool_call_chunk.index,
							'type': 'function',
						}
					elif tool_call_chunk.function.arguments:
						current_tool_call['function']['arguments'] += tool_call_chunk.function.arguments
		if current_tool_call:
			tool_calls.append(current_tool_call)
		if tool_calls:
			if llm_content.strip():
				reasoning_msg_builder = response_builder.create_message_builder(message_type = 'reasoning')
				yield f'data: {reasoning_msg_builder.get_message_data().model_dump_json()}\n\n'
				reasoning_content_builder = reasoning_msg_builder.create_content_builder()
				yield f'data: {reasoning_content_builder.add_text_delta(llm_content).model_dump_json()}\n\n'
				yield f'data: {reasoning_content_builder.complete().model_dump_json()}\n\n'
				yield f'data: {reasoning_msg_builder.complete().model_dump_json()}\n\n'
			messages.append({
				'content': None,
				'role': 'assistant',
				'tool_calls': tool_calls,
			})
			for tool_call in tool_calls:
				plugin_call_msg_builder = response_builder.create_message_builder(message_type = 'plugin_call')
				tool_args = loads(tool_call['function']['arguments'])
				yield f'data: {plugin_call_msg_builder.get_message_data().model_dump_json()}\n\n'
				plugin_call_content_builder = plugin_call_msg_builder.create_content_builder('data')
				yield f'data: {plugin_call_content_builder.add_data_delta({
					'name': tool_call['function']['name'],
					'arguments': dumps(tool_args, ensure_ascii = False),
				}).model_dump_json()}\n\n'
				yield f'data: {plugin_call_content_builder.complete().model_dump_json()}\n\n'
				yield f'data: {plugin_call_msg_builder.complete().model_dump_json()}\n\n'
				content = ''
				try:
					plugin_output_msg_builder = response_builder.create_message_builder(message_type = 'plugin_call_output')
					async with Client('http://127.0.0.1:8080/mcp/') as c:
						LOGGER.warning(f'工具调用（{request_data.session_id}）：{tool_call['function']['name']} - {tool_call['id']}')
						LOGGER.debug(f'参数（{tool_call['id']}）：{tool_args}')
						result = await c.call_tool(tool_call['function']['name'], tool_args)
						for content_item in result.content:
							if isinstance(content_item, TextContent):
								content = dumps(content_item.text, ensure_ascii = False)
								break
							if isinstance(content_item, (AudioContent, ImageContent)):
								content = dumps(content_item.data, ensure_ascii = False)
								break
					yield f'data: {plugin_output_msg_builder.get_message_data().model_dump_json()}\n\n'
					plugin_output_content_builder = plugin_output_msg_builder.create_content_builder(content_type = 'data')
					yield f'data: {plugin_output_content_builder.add_data_delta({
						'name': tool_call['function']['name'],
						'output': content,
					}).model_dump_json()}\n\n'
					yield f'data: {plugin_output_content_builder.complete().model_dump_json()}\n\n'
					yield f'data: {plugin_output_msg_builder.complete().model_dump_json()}\n\n'
					LOGGER.debug(f'输出（{tool_call['id']}）：{content}')
				except Exception as e:
					content = f'Error: {e}'
					LOGGER.debug(f'错误（{tool_call['id']}）：{e}')
				messages.append({
					'content': content,
					'role': 'tool',
					'tool_call_id': tool_call['id'],
				})
			final_response = await client.chat.completions.create(messages = messages, model = MODEL, stream = True)
			final_msg_builder = response_builder.create_message_builder()
			yield f'data: {final_msg_builder.get_message_data().model_dump_json()}\n\n'
			final_content_builder = final_msg_builder.create_content_builder()
			async for chunk in final_response:
				if chunk.choices and len(chunk.choices) > 0:
					choice = chunk.choices[0]
					if choice.delta.content:
						llm_content += choice.delta.content
						yield f'data: {final_content_builder.add_text_delta(choice.delta.content).model_dump_json()}\n\n'
			yield f'data: {final_content_builder.complete().model_dump_json()}\n\n'
			yield f'data: {final_msg_builder.complete().model_dump_json()}\n\n'
		else:
			msg_builder = response_builder.create_message_builder()
			yield f'data: {msg_builder.get_message_data().model_dump_json()}\n\n'
			content_builder = msg_builder.create_content_builder()
			yield f'data: {content_builder.add_text_delta(llm_content).model_dump_json()}\n\n'
			yield f'data: {content_builder.complete().model_dump_json()}\n\n'
			yield f'data: {msg_builder.complete().model_dump_json()}\n\n'
		messages.append({
			'content': llm_content,
			'role': 'assistant',
		})
		yield f'data: {response_builder.completed().model_dump_json()}\n\n'
	except Exception as e:
		error_msg_builder = response_builder.create_message_builder(message_type = 'error')
		error_content_builder = error_msg_builder.create_content_builder()
		yield f'data: {error_content_builder.add_text_delta(f'Error occurred: {str(e)}').model_dump_json()}\n\n'
		yield f'data: {error_content_builder.complete().model_dump_json()}\n\n'
		yield f'data: {error_msg_builder.complete().model_dump_json()}\n\n'
		yield f'data: {response_builder.completed().model_dump_json()}\n\n'
	encoding = get_encoding('o200k_base')
	LOGGER.warning(f'估计消耗 Token（{request_data.session_id}）：{sum(len(encoding.encode(str(value))) for message in messages for value in message.values())}')

@app.post('/process')
def chat(request_data: ChatRequest):
	'''
	Chat interface implementation, supports LLM calls and MCP tool calls

	Core workflow:
	1. Receive user message
	2. Get MCP tool list
	3. Call LLM (with function calling)
	4. If LLM needs to call tools, call MCP tools
	5. Return tool results to LLM
	6. Return final response (conforms to AgentScope ResponseBuilder format)
	'''
	return StreamingResponse(generateResponse(request_data), media_type = 'text/event-stream')

if __name__ == '__main__':
	from uvicorn import run
	run(app, port = 8080)
