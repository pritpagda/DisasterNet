from motor.motor_asyncio import AsyncIOMotorClient
import os

DB_URL = os.getenv("MONGO_DB_URL")

async def connect_to_db(app):
    client = AsyncIOMotorClient(DB_URL)
    app.mongodb = client.artiflora