"""ASGI 入口：uvicorn app:app（阶段一正式入口）。

CLI 运维入口已移至 cli.py：python cli.py <init|ingest|skills|ask|history>
"""
from api.app import app

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run("app:app", host=API_HOST, port=API_PORT)
