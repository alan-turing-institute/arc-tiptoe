"""
Create subset using only queries that have qrels entries
"""

from pathlib import Path


def get_queries_with_qrels(qrels_file: str, max_queries: int = 10) -> list:
    """Get query IDs that have qrels entries"""
    query_ids = set()

    if not Path(qrels_file).exists():
        print(f"❌ Qrels file not found: {qrels_file}")
        return []

    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0].isdigit():
                relevance = int(parts[3])
                if relevance > 0:  # Only relevant documents
                    query_ids.add(parts[0])

                if len(query_ids) >= max_queries:
                    break

    return sorted(list(query_ids))[:max_queries]


def create_subset_with_qrels(
    data_dir: str = "/home/azureuser", subset_queries: int = 10
):
    """Create query subset using only queries with qrels"""

    # Find qrels file
    qrels_locations = [
        f"{data_dir}/msmarco_data/msmarco-docdev-qrels.tsv",
        f"{data_dir}/msmarco_checkpoints/msmarco-docdev-qrels.tsv",
    ]

    qrels_file = None
    for location in qrels_locations:
        if Path(location).exists():
            qrels_file = location
            break

    if not qrels_file:
        print("❌ No qrels file found")
        return None

    print(f"📚 Using qrels file: {qrels_file}")

    # Get queries that have qrels
    query_ids_with_qrels = get_queries_with_qrels(qrels_file, subset_queries)

    if not query_ids_with_qrels:
        print("❌ No queries with qrels found")
        return None

    print(f"✅ Found {len(query_ids_with_qrels)} queries with qrels")
    print(f"   Sample query IDs: {query_ids_with_qrels[:5]}")

    # Find original MSMARCO queries file
    query_file_locations = [
        f"{data_dir}/msmarco_data/msmarco-docdev-queries.tsv",
        f"{data_dir}/msmarco_checkpoints/msmarco-docdev-queries.tsv",
    ]

    original_queries_file = None
    for location in query_file_locations:
        if Path(location).exists():
            original_queries_file = location
            break

    if not original_queries_file:
        print("❌ No original queries file found")
        return None

    # Read original queries and create subset
    queries_dict = {}
    with open(original_queries_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid = parts[0]
                query_text = parts[1]
                queries_dict[qid] = query_text

    # Create quick queries file with only queries that have qrels
    quick_dir = Path(f"{data_dir}/quick_data")
    quick_dir.mkdir(exist_ok=True)

    quick_queries_file = quick_dir / "quick_queries.tsv"

    with open(quick_queries_file, "w", encoding="utf-8") as f:
        for qid in query_ids_with_qrels:
            if qid in queries_dict:
                f.write(f"{qid}\t{queries_dict[qid]}\n")

    print(f"✅ Created {len(query_ids_with_qrels)} queries with guaranteed qrels")
    print(f"   Saved to: {quick_queries_file}")

    return str(quick_queries_file)


if __name__ == "__main__":
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser"
    subset_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    result = create_subset_with_qrels(base_dir, subset_queries)
    if result:
        print(f"\n🎯 Success! Use this query file: {result}")
    else:
        print("\n❌ Failed to create subset")
