import sys
import os
sys.path.append('/fred/oz446/HenryNguyen')

from reranking.reranker import load_reranker, DocumentReranker
from retrieval.retrieval_with_rerank import retrieve_and_rerank, retrieve_chunks
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reranker_standalone():
    print("\n" + "="*80)
    print("TESTING RERANKER STANDALONE")
    print("="*80)
    # Sample documents for testing
    test_documents = [
        {
            "chunk_id": "doc_1",
            "content": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, increased hunger, fatigue, and blurred vision. Type 1 diabetes symptoms often develop quickly, while Type 2 diabetes symptoms may develop more gradually.",
            "metadata": {"source": "diabetes_guide", "type": "symptoms"}
        },
        {
            "chunk_id": "doc_2",
            "content": "Heart disease remains one of the leading causes of death worldwide. Risk factors for heart disease include high blood pressure, high cholesterol levels, smoking, diabetes, obesity, and lack of physical activity. Prevention strategies include maintaining a healthy diet, regular exercise, and avoiding tobacco use.",
            "metadata": {"source": "heart_health", "type": "prevention"}
        },
        {
            "chunk_id": "doc_3",
            "content": "Type 2 diabetes is the most common form of diabetes. Early warning signs include increased thirst and urination, fatigue, blurred vision, slow-healing cuts or bruises, and frequent infections. Many people with type 2 diabetes have no symptoms initially, which is why regular screening is important.",
            "metadata": {"source": "type2_diabetes", "type": "symptoms"}
        },
        {
            "chunk_id": "doc_4",
            "content": "Hypertension, also known as high blood pressure, is often called the 'silent killer' because it typically has no symptoms. However, when symptoms do occur, they may include headaches, shortness of breath, nosebleeds, and chest pain. Regular monitoring is essential for early detection.",
            "metadata": {"source": "hypertension_info", "type": "symptoms"}
        },
        {
            "chunk_id": "doc_5",
            "content": "Managing diabetes involves monitoring blood glucose levels, following a healthy diet, getting regular exercise, and taking medications as prescribed. Patients should work closely with their healthcare team to develop an individualized treatment plan. Regular check-ups are crucial for preventing complications.",
            "metadata": {"source": "diabetes_management", "type": "treatment"}
        }
    ]
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to prevent heart disease?",
        "What are the signs of high blood pressure?"
    ]
    
    # Test both reranker models
    reranker_models = ["bge-reranker-v2-m3", "bge-reranker-v2-gemma"]
    
    for model_name in reranker_models:
        print(f"\n--- Testing {model_name} ---")
        try:
            reranker = load_reranker(model_name)
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                print("-" * 50)
                
                reranked_docs = reranker.rerank_documents(query, test_documents, top_k=3)
                
                for i, doc in enumerate(reranked_docs, 1):
                    print(f"{i}. Score: {doc['rerank_score']:.4f} | ID: {doc['chunk_id']}")
                    print(f"   Content: {doc['content'][:100]}...")
                    print()
                    
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            continue

def test_integrated_pipeline():
    print("\n" + "="*80)
    print("TESTING INTEGRATED PIPELINE")
    print("="*80)
    
    # Configuration
    query = "What are the symptoms of diabetes?"
    model_path = "/fred/oz446/HenryNguyen/EmbeddingModel/MedEmbed-large-v0.1"
    index_path = "/fred/oz446/HenryNguyen/data/hnsw_index.bin"
    
    print(f"Query: {query}")
    print(f"Embedding Model: {model_path}")
    print(f"Index Path: {index_path}")
    
    # Test different configurations
    test_configs = [
        {"reranker": "bge-reranker-v2-m3", "initial_k": 10, "final_k": 5},
        {"reranker": "bge-reranker-v2-gemma", "initial_k": 10, "final_k": 5},
    ]
    
    for config in test_configs:
        print(f"\n--- Configuration: {config} ---")
        
        try:
            # Test the integrated pipeline
            results = retrieve_and_rerank(
                query=query,
                model_path=model_path,
                index_path=index_path,
                reranker_model=config["reranker"],
                initial_top_k=config["initial_k"],
                final_top_k=config["final_k"]
            )
            
            if results:
                print(f"Successfully retrieved and reranked {len(results)} documents:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. Chunk ID: {result.get('chunk_id', 'N/A')}")
                    print(f"   Original Distance: {result.get('distance', 'N/A'):.4f}")
                    print(f"   Rerank Score: {result.get('rerank_score', 'N/A'):.4f}")
                    print(f"   Content: {result.get('content', '')[:150]}...")
                    print()
            else:
                print("No results returned")
                
        except Exception as e:
            logger.error(f"Error testing configuration {config}: {e}")
            continue

def test_comparison():
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs RETRIEVE+RERANK")
    print("="*80)
    
    query = "What are the symptoms of diabetes?"
    model_path = "/fred/oz446/HenryNguyen/EmbeddingModel/MedEmbed-large-v0.1"
    index_path = "/fred/oz446/HenryNguyen/data/hnsw_index.bin"
    
    try:
        # Original retrieval
        print("\n--- ORIGINAL RETRIEVAL (Top 5) ---")
        original_results = retrieve_chunks(query, model_path, index_path, top_k=10)
        
        for i, result in enumerate(original_results, 1):
            print(f"{i}. Chunk ID: {result['chunk_id']}")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Content: {result['content'][:100]}...")
            print()
        
        # Retrieve + Rerank
        print("\n--- RETRIEVE + RERANK (Top 3 from 10) ---")
        reranked_results = retrieve_and_rerank(
            query=query,
            model_path=model_path,
            index_path=index_path,
            reranker_model="bge-reranker-v2-m3",
            initial_top_k=10,
            final_top_k=5
        )
        
        for i, result in enumerate(reranked_results, 1):
            print(f"{i}. Chunk ID: {result['chunk_id']}")
            print(f"   Original Distance: {result['distance']:.4f}")
            print(f"   Rerank Score: {result.get('rerank_score', 'N/A'):.4f}")
            print(f"   Content: {result['content'][:100]}...")
            print()
            
    except Exception as e:
        logger.error(f"Error in comparison test: {e}")

def main():
    print("Starting Reranking Tests...")
    
    # Check if required paths exist
    required_paths = [
        "/fred/oz446/HenryNguyen/reranker/bge-reranker-v2-m3",
        "/fred/oz446/HenryNguyen/reranker/bge-reranker-v2-gemma",
        "/fred/oz446/HenryNguyen/EmbeddingModel/MedEmbed-large-v0.1",
        "/fred/oz446/HenryNguyen/data/hnsw_index.bin"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("WARNING: The following required paths are missing:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nSome tests may fail. Please ensure all models and data are available.")
    
    # Run tests
    try:
        test_reranker_standalone()
    except Exception as e:
        logger.error(f"Standalone reranker test failed: {e}")
    
    try:
        test_integrated_pipeline()
    except Exception as e:
        logger.error(f"Integrated pipeline test failed: {e}")
    
    try:
        test_comparison()
    except Exception as e:
        logger.error(f"Comparison test failed: {e}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
