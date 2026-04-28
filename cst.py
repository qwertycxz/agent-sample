from logging import getLogger
from os import environ
from typing import Any

from agentscope.message import TextBlock
from agentscope.rag import DocMetadata, Document, KnowledgeBase
from aiohttp import ClientSession

class CSTKnowledgeBase(KnowledgeBase):
	LOGGER = getLogger('科技云知识库')

	def __init__(self, id: str, name: str):
		'''
		初始化科技云知识库
		Args:
			id: 知识库ID，用于 API 调用
			name: 知识库名称，用于日志记录
		'''
		self.id = id
		self.name = name

	async def retrieve(self, query: str, limit: int = 5, score_threshold: float | None = None, **kwargs: Any):
		self.LOGGER.info(f'查询 {self.name}: {query}')
		async with ClientSession() as s:
			async with s.post('https://kb.cstcloud.cn/api/user/resource/kbs/dify/retrieval', headers = {
				'Authorization': f'Bearer {environ['CST_KB_KEY']}',
			}, json = {
				'knowledge_id': self.id,
				'query': query,
				'retrieval_setting': {
					'score_threshold': score_threshold,
					'top_k': limit,
				},
			}) as p:
				return [Document(DocMetadata(TextBlock(text = f'{record['content']}', type = 'text'), record['title'], 0, 0), score = record['score']) for record in (await p.json())['records']]
