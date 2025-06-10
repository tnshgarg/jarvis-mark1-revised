import asyncio
from src.mark1.storage.database import get_db_session
from src.mark1.storage.repositories.agent_repository import AgentRepository

async def check_db():
    async with get_db_session() as session:
        agent_repo = AgentRepository()
        agents = await agent_repo.list_all(session)
        print(f'Found {len(agents)} agents in database:')
        for agent in agents:
            print(f'  - {agent.name} ({agent.id})')
            print(f'    File: {agent.file_path}')
            print(f'    Status: {agent.status}')
            print(f'    Type: {agent.agent_type}')
            print(f'    Framework: {agent.framework_version}')
            print(f'    Capabilities: {agent.capabilities}')
            print()

if __name__ == "__main__":
    asyncio.run(check_db()) 