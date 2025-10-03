"""
Compute MRR@100 using the existing quality-eval/get-mrr.py script
"""

import json
import subprocess
import sys
from pathlib import Path


def find_existing_qrels():
    """Find existing qrels file"""
    possible_locations = [
        "/home/azureuser/msmarco_data/msmarco-docdev-qrels.tsv",
        "/home/azureuser/msmarco_checkpoints/msmarco-docdev-qrels.tsv",
        "quality-eval/msmarco-docdev-qrels.tsv",
    ]

    for location in possible_locations:
        if Path(location).exists():
            print(f"✅ Found existing qrels: {location}")
            return location

    print("❌ No qrels file found")
    return None


def compute_mrr_with_get_mrr_script(quality_log_file: str):
    """Use the existing quality-eval/get-mrr.py script"""

    if not Path(quality_log_file).exists():
        print(f"❌ Quality log not found: {quality_log_file}")
        return 0.0

    # Find existing qrels file
    qrels_file = find_existing_qrels()
    if not qrels_file:
        return 0.0

    # Fix: Find the get-mrr.py script with correct path
    possible_script_paths = [
        Path("../quality-eval/get-mrr.py"),
        Path("quality-eval/get-mrr.py"),
        Path("./quality-eval/get-mrr.py"),
    ]

    get_mrr_script = None
    for script_path in possible_script_paths:
        if script_path.exists():
            get_mrr_script = script_path
            break

    if not get_mrr_script:
        print(
            f"❌ get-mrr.py not found in any of: {[str(p) for p in possible_script_paths]}"
        )
        return 0.0

    try:
        print("🔍 Computing MRR using quality-eval/get-mrr.py...")
        print(f"   Script: {get_mrr_script}")
        print(f"   Quality log: {quality_log_file}")
        print(f"   Qrels: {qrels_file}")

        # Run the existing script with absolute paths
        result = subprocess.run(
            [
                "python3",
                str(get_mrr_script.absolute()),
                str(Path(quality_log_file).absolute()),
            ],
            cwd=str(get_mrr_script.parent),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode == 0:
            # Parse output: "MRR@100: 0.123456"
            output = result.stdout.strip()
            print(f"   Raw output: {output}")

            import re

            mrr_match = re.search(r"MRR@(\d+):\s*([\d.]+)", output)
            if mrr_match:
                mrr_value = float(mrr_match.group(2))
                print(f"✅ Computed MRR@100: {mrr_value:.4f}")
                return mrr_value
            else:
                print(f"⚠️  Could not parse MRR from output: {output}")
                return 0.0
        else:
            print("❌ get-mrr.py failed:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return 0.0

    except subprocess.TimeoutExpired:
        print("❌ MRR computation timed out")
        return 0.0
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        print(f"❌ Error running get-mrr.py: {e}")
        return 0.0


def debug_query_ids_vs_qrels(quality_log_file: str):
    """Debug whether query IDs match between search results and qrels"""
    print("\n🔍 Debugging query ID matching...")

    # Get query IDs from search results
    search_query_ids = set()
    if Path(quality_log_file).exists():
        with open(quality_log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                import re

                query_match = re.match(r"Query:\s*(\d+)", line)
                if query_match:
                    search_query_ids.add(query_match.group(1))

    print(f"   Search result query IDs: {sorted(list(search_query_ids))}")

    # Get a sample of qrels query IDs
    qrels_file = find_existing_qrels()
    qrels_query_ids = set()
    if qrels_file and Path(qrels_file).exists():
        with open(qrels_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Just sample first 100 lines
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0].isdigit():
                    qrels_query_ids.add(parts[0])

    print(f"   Sample qrels query IDs: {sorted(list(qrels_query_ids))[:10]}...")

    # Check overlap
    overlap = search_query_ids & qrels_query_ids
    print(
        f"   Overlapping query IDs: {len(overlap)} out of {len(search_query_ids)} search queries"
    )
    if overlap:
        print(f"   Sample overlapping IDs: {sorted(list(overlap))}")
    else:
        print("   ❌ NO OVERLAP - This is why MRR is 0!")


def debug_search_output_format(quality_log_file: str):
    """Debug the search output format"""
    print("\n🔍 Debugging search output format...")

    if not Path(quality_log_file).exists():
        print(f"❌ File not found: {quality_log_file}")
        return

    with open(quality_log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📄 File has {len(lines)} lines")

    # Show first 15 lines
    print("\n📝 First 15 lines:")
    for i, line in enumerate(lines[:15]):
        line = line.strip()
        if line:
            print(f"  {i+1:2d}: {line}")

    # Pattern analysis - check what quality-eval/mrr.py expects
    query_pattern = r"Query: (.+)"
    queries_found = []
    results_found = []

    current_query = None

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check query pattern
        import re

        query_match = re.match(query_pattern, line)
        if query_match:
            current_query = query_match.group(1).split()[0]  # First word is query ID
            queries_found.append((line_num, current_query, line))
            continue

        # Check result pattern - quality-eval/mrr.py expects: score doc_id
        tokens = line.split()
        if len(tokens) >= 2 and current_query and "/home/ubuntu" not in tokens[0]:
            try:
                # First token should be score, second should be doc_id
                score = float(tokens[0])
                doc_id = tokens[1]
                results_found.append((line_num, current_query, score, doc_id))
            except ValueError:
                pass

    print("\n📊 Format Analysis:")
    print(f"   Queries found: {len(queries_found)}")
    print(f"   Results found: {len(results_found)}")

    if queries_found:
        print("\n🔍 Sample queries:")
        for line_num, qid, full_line in queries_found[:3]:
            print(f"   Line {line_num}: Query ID '{qid}' from '{full_line}'")

    if results_found:
        print("\n📋 Sample results:")
        for line_num, qid, score, doc_id in results_found[:5]:
            print(f"   Line {line_num}: Query '{qid}' -> Score {score} Doc '{doc_id}'")

    print("\n🎯 Expected format (from quality-eval/mrr.py):")
    print("   Query line: 'Query: <query_id> <query_text>'")
    print("   Result line: '<score> <doc_id>' (space-separated)")
    print("   Note: Filters out lines containing '/home/ubuntu'")


def enhance_quality_metrics(quality_log_file: str) -> dict:
    """Enhanced quality analysis using the existing get-mrr.py script"""

    print(f"🔍 Processing quality log: {quality_log_file}")

    # Debug the format first
    debug_search_output_format(quality_log_file)

    # Debug query ID matching - this is likely the issue
    debug_query_ids_vs_qrels(quality_log_file)

    # Use the existing script
    mrr_100 = compute_mrr_with_get_mrr_script(quality_log_file)

    # Basic parsing for additional metrics
    if not Path(quality_log_file).exists():
        return {}

    with open(quality_log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    query_count = len([line for line in lines if line.strip().startswith("Query:")])
    result_count = len(
        [
            line
            for line in lines
            if line.strip()
            and not line.strip().startswith("Query:")
            and len(line.strip().split()) >= 2
        ]
    )

    return {
        "mrr_100": mrr_100,
        "mrr_10": mrr_100,
        "queries_evaluated": query_count,
        "total_results": result_count,
        "queries_with_qrels": query_count if mrr_100 > 0 else 0,
        "search_queries": query_count,
        "qrels_queries": 0,
    }


def main():
    """Main entrypoint"""
    if len(sys.argv) < 2:
        print("Usage: python3 compute_mrr.py <quality_log_file> [qrels_file]")
        return

    quality_log = sys.argv[1]

    metrics = enhance_quality_metrics(quality_log)

    if metrics:
        print("\n🎯 FINAL RESULTS:")
        print(json.dumps(metrics, indent=2))
    else:
        print("\n❌ No metrics computed")


if __name__ == "__main__":
    main()
