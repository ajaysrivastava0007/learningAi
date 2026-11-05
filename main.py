
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

pdf_path = Path("documents/CV.pdf")

if not pdf_path.exists():
    print("PDF not found!")
else:
    reader = PdfReader(str(pdf_path))

    text = "\n\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # print(f"Total chunks: {len(chunks)}")

    # for i, chunk in enumerate(chunks, start=1):
    #     print(f"\n--- Chunk {i} ---")
    #     print(chunk)

if not chunks:
        print("No readable text found in the PDF.")
else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)

        # print(f"Total chunks: {len(chunks)}")
        # print(f"Embedding shape: {embeddings.shape}")
        # print(f"\nFirst chunk:\n{chunks[0]}")
        # print(f"\nFirst embedding (first 10 values):")
        # print(embeddings[0][:10])
        #print(embeddings)

        client = chromadb.PersistentClient(path="./chroma_db")

        collection = client.get_or_create_collection(
            name="pdf_documents"
        )

        collection.upsert(
            ids=[f"sample_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        # print(f"Total chunks stored: {len(chunks)}")
        # print(f"Total records in ChromaDB: {collection.count()}")

        question = "who is ajay working?"

query_embedding = model.encode([question])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=min(3, collection.count())
)

# print("\nRelevant chunks:")

# for chunk in results["documents"][0]:
#     print(f"\n{chunk}")

print(results)
# print(results)