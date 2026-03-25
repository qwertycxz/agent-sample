from logging import getLogger
from typing import Any

from agentscope.message import TextBlock
from agentscope.rag import DocMetadata, Document, KnowledgeBase
from agentscope.tool import ToolResponse
from aiohttp import ClientSession

class CSTKnowledgeBase(KnowledgeBase):
	LOGGER = getLogger('科技云知识库')

	def __init__(self, authorization: str, id: str, name: str):
		'''
		初始化科技云知识库
		Args:
			authorization: 访问科技云知识库的授权令牌，格式为`Bearer ak-ou-aB3kF9mN2pQ7rS5tU8vW1xY4zA6cE0gH`
			id: 知识库ID
			name: 知识库名称
		'''
		self.authorization = authorization
		self.id = id
		self.name = name

	async def retrieve(self, query: str, limit: int = 5, score_threshold: float | None = None, **kwargs: Any):
		self.LOGGER.info(f'查询 {self.name}: {query}')
		async with ClientSession() as s:
			async with s.post('https://kb.cstcloud.cn/api/user/resource/kbs/dify/retrieval', headers = {
				'Authorization': self.authorization,
			}, json = {
				'knowledge_id': self.id,
				'query': query,
				'retrieval_setting': {
					'score_threshold': score_threshold,
					'top_k': limit,
				},
			}) as p:
				return [Document(DocMetadata(TextBlock(text = f'{record['content']}', type = 'text'), record['title'], 0, 0), score = record['score']) for record in (await p.json())['records']]


# 本地工具调用函数，必须提供详细的 docstring（就是这些三引号括起来的注释），以便智能体能够正确使用工具
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
