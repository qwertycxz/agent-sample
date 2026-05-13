#!/usr/bin/env python
from asyncio import TimerHandle, get_event_loop
from base64 import b64decode, b64encode
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from http.client import CONFLICT, FORBIDDEN, NOT_FOUND, UNAUTHORIZED
from os import urandom
from typing import Annotated, TypedDict

from agentscope_runtime.engine import AgentApp
from fastapi import Body, Depends, HTTPException, Request, Response
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, Session, mapped_column, sessionmaker

def hmac(password: bytes, salt: bytes):
	'''
	计算登录用密码哈希值的全局函数
	Args:
		password (bytes): 密码
		salt (bytes): 盐值
	Returns:
		密码哈希值
	'''
	return pbkdf2_hmac('blake2b', password, salt, 100000)

class Base(DeclarativeBase, MappedAsDataclass):
	'''
	SQLAlchemy 基类，启用数据类功能
	'''
	pass

class SessionStore(TypedDict):
	'''
	暂存在内存内的会话信息
	'''
	expire: TimerHandle
	user: User

class User(Base):
	'''
	用户模型
	'''
	__tablename__ = 'users'
	id: Mapped[int] = mapped_column(comment='ID', init=False, primary_key=True)
	name: Mapped[str] = mapped_column(String(100), comment='用户名', nullable=False, unique=True)
	password: Mapped[bytes] = mapped_column(comment='密码', nullable=False)
	salt: Mapped[bytes] = mapped_column(comment='密码盐值', nullable=False)
	admin: Mapped[bool] = mapped_column(comment='是否管理员', nullable=False, default=False)

engine = create_engine('mysql+mysqlconnector://root@127.0.0.1/tender')
makeDatabase = sessionmaker(engine)

def getDatabase():
	'''
	从池中获取一个连接，生成器关闭时自动归还连接

	Yields:
		一个数据库连接
	'''
	database = makeDatabase()
	try:
		yield database
	finally:
		database.close()

sessions: dict[bytes, SessionStore] = {}
def getUser(request: Request):
	'''
	从请求的 Cookie 中获取会话信息，验证后返回用户对象

	Args:
		request: 当前请求对象

	Raises:
		HTTPException: 如果没有有效的会话信息，抛出 401

	Returns:
		和数据库分离的用户对象
	'''
	if 'session' not in request.cookies:
		raise HTTPException(UNAUTHORIZED, '您尚未登录！')
	session = b64decode(request.cookies['session'])
	if session in sessions:
		return sessions[session]['user']
	raise HTTPException(UNAUTHORIZED, '请重新登录！')

app = AgentApp()

@app.delete('/auth')
def logout(user: User = Depends(getUser)):
	'''
	退出登录

	Args:
		user: 当前用户对象
	'''
	for token, session in list(sessions.items()):
		if session['user'].id == user.id:
			session['expire'].cancel()
			sessions.pop(token, None)

@app.post('/auth')
async def login(name: Annotated[str, Body()], password: Annotated[bytes, Body()], response: Response, database: Session = Depends(getDatabase)):
	'''
	登录

	Args:
		name: 用户名
		password: 密码
		response: 当前响应对象
		database: 数据库连接

	Raises:
		HTTPException: 如果用户名不存在，抛出 404；如果密码错误，抛出 401

	Returns:
		一个包含用户 ID 和管理员状态的字典
	'''
	user = database.execute(select(User).where(User.name == name)).scalar()
	if not user:
		raise HTTPException(NOT_FOUND, '用户不存在！')

	if not compare_digest(hmac(password, user.salt), user.password):
		raise HTTPException(UNAUTHORIZED, '密码错误！')

	# token = str(uuid4())
	token = urandom(16)
	sessions[token] = {
		'expire': get_event_loop().call_later(1000000, lambda: sessions.pop(token, None)),
		'user': user
	}

	response.set_cookie('session', b64encode(token).decode())
	return {
		'admin': user.admin,
		'id': user.id
	}

@app.delete('/user')
def deleteUser(name: Annotated[str, Body(embed=True)], database: Session = Depends(getDatabase), user: User = Depends(getUser)):
	'''
	删除用户

	Args:
		name: 用户名
		database: 数据库连接
		user: 当前用户对象

	Raises:
		HTTPException: 如果当前用户不是管理员，抛出 403；如果欲删除用户不存在，抛出 404
	'''
	if not user.admin:
		raise HTTPException(FORBIDDEN, '权限不足！')

	deleted = database.execute(select(User).where(User.name == name)).scalar()
	if not deleted:
		raise HTTPException(NOT_FOUND, '用户不存在！')

	database.delete(deleted)
	database.commit()
	logout(deleted)

@app.get('/user')
def listUsers(database: Session = Depends(getDatabase), user: User = Depends(getUser)):
	'''
	列出所有用户

	Args:
		database: 数据库连接
		user: 当前用户对象

	Raises:
		HTTPException: 如果当前用户不是管理员，抛出 403

	Returns:
		一个包含所有用户 ID、用户名和管理员状态的列表
	'''
	if not user.admin:
		raise HTTPException(FORBIDDEN, '权限不足！')
	return database.execute(select(User.admin, User.id, User.name)).mappings().all()

@app.post('/user')
def register(name: Annotated[str, Body()], password: Annotated[bytes, Body()], admin: Annotated[bool, Body()] = False, database: Session = Depends(getDatabase), user: User = Depends(getUser)):
	'''
	注册用户

	Args:
		name: 用户名
		password: 密码
		admin: 是否为管理员
		database: 数据库连接
		user: 当前用户对象

	Raises:
		HTTPException: 如果当前用户不是管理员或用户已存在，抛出 409

	Returns:
		一个包含新用户 ID 的字典
	'''
	if not user.admin or database.execute(select(User).where(User.name == name)).scalar():
		raise HTTPException(CONFLICT, '用户已存在！')

	salt = urandom(16)
	registered = User(name, hmac(password, salt), salt, admin)

	database.add(registered)
	database.commit()

	return {
		'id': registered.id
	}

@app.put('/user')
def updateUser(name: Annotated[str, Body()], admin: Annotated[bool | None, Body()] = None, password: Annotated[bytes | None, Body()] = None, database: Session = Depends(getDatabase), user: User = Depends(getUser)):
	'''
	更新用户信息

	Args:
		name: 用户名
		admin: 管理员状态
		password: 密码
		database: 数据库连接
		user: 当前用户对象

	Raises:
		HTTPException: 如果当前用户不是管理员且尝试修改其他用户的信息，抛出 403；如果欲修改用户不存在，抛出 404

	Returns:
		一个包含更新后用户 ID 和管理员状态的字典
	'''
	if not user.admin and (name != user.name or admin is not None):
		raise HTTPException(FORBIDDEN, '这不是你自己！')

	updated = database.execute(select(User).where(User.name == name)).scalar()
	if not updated:
		raise HTTPException(NOT_FOUND, '用户不存在！')

	if admin is not None:
		updated.admin = admin

	if password:
		salt = urandom(16)
		updated.password = hmac(password, salt)
		updated.salt = salt

	database.commit()
	logout(updated)

	return {
		'admin': updated.admin,
		'id': updated.id
	}

if __name__ == '__main__':
	Base.metadata.create_all(engine)
	app.run()
