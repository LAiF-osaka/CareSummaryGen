import chromadb

client = chromadb.PersistentClient(path="../vectorDB")

collection = client.get_collection(name="medical_record")
print(collection.count())
print(collection.peek())