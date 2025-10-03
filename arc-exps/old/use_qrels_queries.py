"""
Create subset using only queries that have qrels entries
"""

from pathlib import Path


def find_qrels_file(search_dir: str = "/home/azureuser") -> str:
    """Find the qrels file in common locations"""
    possible_locations = [
        f"{search_dir}/msmarco_data/msmarco-docdev-qrels.tsv",
        f"{search_dir}/msmarco_checkpoints/msmarco-docdev-qrels.tsv",
        "quality-eval/msmarco-docdev-qrels.tsv",
        "/home/azureuser/msmarco_data/msmarco-docdev-qrels.tsv",
        "/home/azureuser/msmarco_checkpoints/msmarco-docdev-qrels.tsv",
    ]

    for location in possible_locations:
        if Path(location).exists():
            print(f"✅ Found qrels file: {location}")
            return location

    print(f"❌ No qrels file found in any of: {possible_locations}")
    return None


def get_queries_with_qrels(qrels_file: str, max_queries: int = 10) -> list:
    """Get query IDs that have qrels entries"""
    query_ids = set()

    if not Path(qrels_file).exists():
        print(f"❌ Qrels file not found: {qrels_file}")
        return []

    print(f"📚 Reading qrels from: {qrels_file}")

    with open(qrels_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    relevance = int(parts[3])
                    if relevance > 0:  # Only relevant documents
                        query_ids.add(parts[0])

                    if len(query_ids) >= max_queries:
                        break
                except ValueError:
                    continue

            # Show progress every 10000 lines
            if line_num % 10000 == 0 and line_num > 0:
                print(
                    f"   Processed {line_num} lines, found {len(query_ids)} unique queries"
                )

    return sorted(list(query_ids))[:max_queries]


def create_subset_with_qrels(
    search_dir: str = "/home/azureuser", max_subset_queries: int = 10
):
    """Create query subset using only queries with qrels"""

    # Find qrels file
    qrels_file = find_qrels_file(search_dir)

    if not qrels_file:
        print("❌ No qrels file found")
        return None

    print(f"📚 Using qrels file: {qrels_file}")

    # Get queries that have qrels
    query_ids_with_qrels = get_queries_with_qrels(qrels_file, max_subset_queries)

    if not query_ids_with_qrels:
        print("❌ No queries with qrels found")
        return None

    print(f"✅ Found {len(query_ids_with_qrels)} queries with qrels")
    print(f"   Sample query IDs: {query_ids_with_qrels[:5]}")

    # Find original MSMARCO queries file
    query_file_locations = [
        f"{search_dir}/msmarco_data/msmarco-docdev-queries.tsv",
        f"{search_dir}/msmarco_checkpoints/msmarco-docdev-queries.tsv",
        "/home/azureuser/msmarco_data/msmarco-docdev-queries.tsv",
        "/home/azureuser/msmarco_checkpoints/msmarco-docdev-queries.tsv",
    ]

    original_queries_file = None
    for location in query_file_locations:
        if Path(location).exists():
            original_queries_file = location
            print(f"✅ Found queries file: {original_queries_file}")
            break

    if not original_queries_file:
        print("❌ No original queries file found")
        print("💡 Download with:")
        print(
            "   wget -O /home/azureuser/msmarco_checkpoints/msmarco-docdev-queries.tsv \\"
        )
        print(
            "     'https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv'"
        )
        return None

    # Read original queries and create subset
    queries_dict = {}
    print(f"📖 Reading queries from: {original_queries_file}")

    with open(original_queries_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qid = parts[0]
                query_text = parts[1]
                queries_dict[qid] = query_text

            # Show progress
            if line_num % 50000 == 0 and line_num > 0:
                print(f"   Processed {line_num} query lines")

    print(f"📊 Loaded {len(queries_dict)} total queries")

    # Create quick queries file with only queries that have qrels
    quick_dir = Path(f"{search_dir}/quick_data")
    quick_dir.mkdir(exist_ok=True)

    quick_queries_file = quick_dir / "quick_queries.tsv"

    matched_queries = 0
    with open(quick_queries_file, "w", encoding="utf-8") as f:
        for qid in query_ids_with_qrels:
            if qid in queries_dict:
                f.write(f"{qid}\t{queries_dict[qid]}\n")
                matched_queries += 1

    print(f"✅ Created {matched_queries} queries with guaranteed qrels")
    print(f"   Saved to: {quick_queries_file}")

    if matched_queries == 0:
        print("❌ No queries matched between qrels and query file!")
        print(f"   Qrels query IDs (first 5): {query_ids_with_qrels[:5]}")
        print(f"   Query file query IDs (first 5): {list(queries_dict.keys())[:5]}")

    return str(quick_queries_file) if matched_queries > 0 else None


if __name__ == "__main__":
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/azureuser"
    subset_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    result = create_subset_with_qrels(base_dir, subset_queries)
    if result:
        print(f"\n🎯 Success! Use this query file: {result}")
    else:
        print("\n❌ Failed to create subset")
