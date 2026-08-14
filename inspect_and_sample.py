"""
inspect_and_sample.py — Sample real rows from hintrain.parquet
Extract passages and matching queries to construct a real in-domain test set.
"""
import pyarrow.parquet as pq
import json

pf = pq.ParquetFile("data/parquet_cache/hintrain.parquet")

batch_iter = pf.iter_batches(
    batch_size=100,
    columns=["query_id", "query", "Answer", "Eng_Query", "Eng_Answer", "passages"],
)

batch = next(batch_iter)
print(f"Num rows in batch: {batch.num_rows}")

sample_queries = []
sample_passages = []

for i in range(min(20, batch.num_rows)):
    qid = batch.column("query_id")[i].as_py()
    hi_query = batch.column("query")[i].as_py()
    hi_ans = batch.column("Answer")[i].as_py()
    en_query = batch.column("Eng_Query")[i].as_py()
    en_ans = batch.column("Eng_Answer")[i].as_py()
    
    row_passages = batch.column("passages")[i].as_py()
    hi_passages = row_passages.get("Translated_passages", [])
    en_passages = row_passages.get("English_passages", [])
    is_sel = row_passages.get("is_selected", [])
    
    # Check if there is a selected (relevant) passage
    selected_indices = [idx for idx, s in enumerate(is_sel) if s == 1]
    
    if selected_indices and hi_ans and hi_ans != "कोई उत्तर नहीं मिला।":
        sel_idx = selected_indices[0]
        sample_queries.append({
            "qid": qid,
            "hi_query": hi_query,
            "hi_answer": hi_ans,
            "en_query": en_query,
            "en_answer": en_ans,
            "sel_hi_passage": hi_passages[sel_idx] if sel_idx < len(hi_passages) else "",
            "sel_en_passage": en_passages[sel_idx] if sel_idx < len(en_passages) else "",
        })

print(f"Found {len(sample_queries)} high-quality queries with answers:")
for q in sample_queries[:5]:
    print(f"\n[QID {q['qid']}]")
    print(f"  HI Query: {q['hi_query']}")
    print(f"  HI Answer: {q['hi_answer'][:80]}...")
    print(f"  EN Query: {q['en_query']}")
    print(f"  EN Answer: {q['en_answer'][:80]}...")
