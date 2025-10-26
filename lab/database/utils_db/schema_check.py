from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["medical_rag_db"]

chunk_info = db.command("listCollections", filter={"name": "chunk"})
print(chunk_info)

client.close()