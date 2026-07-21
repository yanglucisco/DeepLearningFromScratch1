import os
import asyncio

# 跳过扩展下载（国内网络超时）
os.environ["BROWSER_USE_DISABLE_EXTENSIONS"] = "1"

from browser_use import Agent
from browser_use.llm.deepseek.chat import ChatDeepSeek

# 1. 初始化 DeepSeek 模型（browser-use 原生支持）
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-5055cc3e11c44eccb56b730107cdc5e8",  # 替换为你的 DeepSeek API Key
    temperature=0.1,
)

# 2. 创建 Agent
agent = Agent(
    task="""打开 http://http://192.168.20.1/，在用户名输入框中输入xiaoshou2，然后等待我手动输入密码后，然后点击登录按钮。
    登录成功后，点击左侧的待办，然后点击待办审核，然后点击第一行数据最右侧的办理，然后点击通过按钮，然后输入审批意见：同意，然后点击通过""",
    llm=llm,
    headless=False,
    verbose=True,
)

# 3. 运行 Agent
async def main():
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
