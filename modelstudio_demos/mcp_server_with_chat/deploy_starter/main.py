from agentscope_runtime.engine.helpers.agent_api_builder import ResponseBuilder
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from json import dumps, loads
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageFunctionToolCallParam, ChatCompletionMessageParam, ChatCompletionToolUnionParam
from pydantic import BaseModel
from typing import Any
from deploy_starter.mcp_server import (
	call_mcp_tool,
	convert_mcp_tools_to_openai_format,
	list_mcp_tools,
	mcp,
)

MODEL = 'qwen-plus'

mcp_asgi_app = mcp.streamable_http_app('/')

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
	'''Application lifecycle management - integrate MCP application's lifespan'''
	async with mcp_asgi_app.router.lifespan_context(fastapi_app):
		yield

app = FastAPI(lifespan = lifespan)

app.mount('/mcp', mcp_asgi_app)

@app.get('/')
def readRoot():
	return 'OK'

@app.get('/health')
def healthCheck():
	return 'OK'

class ContentItem(BaseModel):
	data: dict[str, Any] | None = None
	status: str | None = None
	text: str | None = None
	type: str
	class Config:
		extra = 'allow'

class MessageItem(BaseModel):
	content: list[ContentItem] | None = None
	role: str
	type: str | None = None
	class Config:
		extra = 'allow'

def getMessage(msg: MessageItem) -> ChatCompletionMessageParam | None:
	if not msg.content:
		return
	content_text = ''.join((content_item.text for content_item in msg.content if content_item.type == 'text' and content_item.text))
	if not content_text:
		return
	if msg.role == 'user':
		return {
			'role': 'user',
			'content': content_text,
		}
	if msg.role == 'assistant' and msg.type == 'message':
		return {
			'role': 'assistant',
			'content': content_text,
		}

class ChatRequest(BaseModel):
	input: list[MessageItem]
	session_id: str
	stream: bool | None = True

class ToolCall(ChatCompletionMessageFunctionToolCallParam):
	index: int

async def generateResponse(request_data: ChatRequest):
	'''Generate streaming response - conforms to Bailian Response/Message/Content architecture'''
	client = AsyncOpenAI(api_key = 'sk-dc3f5a4ae87047b6a6c46a0cbcb4187e', base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1')
	messages: list[ChatCompletionMessageParam] = [message for msg in request_data.input if (message := getMessage(msg))]
	response_builder = ResponseBuilder(request_data.session_id, f'resp_{request_data.session_id}')
	yield f'data: {response_builder.created().model_dump_json()}\n\n'
	yield f'data: {response_builder.in_progress().model_dump_json()}\n\n'
	openai_tools: list[ChatCompletionToolUnionParam] = []
	try:
		openai_tools = convert_mcp_tools_to_openai_format(await list_mcp_tools())
	except Exception as e:
		print(f'Failed to get MCP tools: {e}')
	try:
		response: AsyncStream[ChatCompletionChunk]
		if openai_tools:
			response = await client.chat.completions.create(messages = messages, model = MODEL, stream = True, tools = openai_tools)
		else:
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
							'index': tool_call_chunk.index,
							'id': tool_call_chunk.id or '',
							'type': 'function',
							'function': {
								'name': tool_call_chunk.function.name or '',
								'arguments': tool_call_chunk.function.arguments or '',
							},
						}
					elif tool_call_chunk.function.arguments:
						current_tool_call['function']['arguments'] += tool_call_chunk.function.arguments
		if current_tool_call:
			tool_calls.append(current_tool_call)
		if tool_calls:
			if llm_content.strip():
				reasoning_msg_builder = (response_builder.create_message_builder(message_type = 'reasoning'))
				yield f'data: {reasoning_msg_builder.get_message_data().model_dump_json()}\n\n'
				reasoning_content_builder = reasoning_msg_builder.create_content_builder()
				yield f'data: {reasoning_content_builder.add_text_delta(llm_content).model_dump_json()}\n\n'
				yield f'data: {reasoning_content_builder.complete().model_dump_json()}\n\n'
				yield f'data: {reasoning_msg_builder.complete().model_dump_json()}\n\n'
			messages.append({
				'role': 'assistant',
				'content': None,
				'tool_calls': tool_calls,
			})
			for tool_call in tool_calls:
				plugin_call_msg_builder = response_builder.create_message_builder(message_type = 'plugin_call')
				tool_args = loads(tool_call['function']['arguments'])
				tool_name = tool_call['function']['name']
				yield f'data: {plugin_call_msg_builder.get_message_data().model_dump_json()}\n\n'
				plugin_call_content_builder = plugin_call_msg_builder.create_content_builder('data')
				yield f'data: {plugin_call_content_builder.add_data_delta({
					'name': tool_name,
					'arguments': dumps(tool_args, ensure_ascii = False),
				}).model_dump_json()}\n\n'
				yield f'data: {plugin_call_content_builder.complete().model_dump_json()}\n\n'
				yield f'data: {plugin_call_msg_builder.complete().model_dump_json()}\n\n'
				content = ''
				try:
					plugin_output_msg_builder = response_builder.create_message_builder(message_type = 'plugin_call_output')
					tool_result = await call_mcp_tool(tool_name, tool_args)
					yield f'data: {plugin_output_msg_builder.get_message_data().model_dump_json()}\n\n'
					if tool_result:
						content = dumps(tool_result, ensure_ascii = False)
					plugin_output_content_builder = plugin_output_msg_builder.create_content_builder(content_type = 'data')
					yield f'data: {plugin_output_content_builder.add_data_delta({
						'name': tool_name,
						'output': content,
					}).model_dump_json()}\n\n'
					yield f'data: {plugin_output_content_builder.complete().model_dump_json()}\n\n'
					yield f'data: {plugin_output_msg_builder.complete().model_dump_json()}\n\n'
				except Exception as e:
					print(f'Tool call failed: {e}')
					content = f'Error: {e}'
				messages.append({
					'role': 'tool',
					'tool_call_id': tool_call['id'],
					'content': content,
				})
			final_response = await client.chat.completions.create(messages = messages, model = MODEL, stream = True)
			final_msg_builder = response_builder.create_message_builder()
			yield f'data: {final_msg_builder.get_message_data().model_dump_json()}\n\n'
			final_content_builder = final_msg_builder.create_content_builder()
			async for chunk in final_response:
				if chunk.choices and len(chunk.choices) > 0:
					choice = chunk.choices[0]
					if choice.delta.content:
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
		yield f'data: {response_builder.completed().model_dump_json()}\n\n'
	except Exception as e:
		print(f'Chat interface error: {e}')
		error_msg_builder = response_builder.create_message_builder(message_type = 'error')
		error_content_builder = error_msg_builder.create_content_builder()
		yield f'data: {error_content_builder.add_text_delta(f'Error occurred: {str(e)}').model_dump_json()}\n\n'
		yield f'data: {error_content_builder.complete().model_dump_json()}\n\n'
		yield f'data: {error_msg_builder.complete().model_dump_json()}\n\n'
		yield f'data: {response_builder.completed().model_dump_json()}\n\n'

@app.post('/process')
async def chat(request_data: ChatRequest):
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
	run('deploy_starter.main:app', port = 8080)
