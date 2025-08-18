"""
Check if a URL is valid and accessible for a given query result
"""

# import glob
import gzip

# import re
import sys


def find_url_by_docid(
    doc_id,
    collection_file="/home/azureuser/.ir_datasets/msmarco-document/collection.tsv.gz",
):
    """Find URL for a given document ID from MS MARCO collection."""

    try:
        # Open the gzipped file
        with gzip.open(collection_file, "rt", encoding="utf-8") as f:
            current_doc = {}
            in_text_section = False
            text_lines = []

            for line in f:
                line = line.strip()

                # Start of a new document
                if line == "<DOC>":
                    current_doc = {}
                    in_text_section = False
                    text_lines = []

                # Document number
                elif line.startswith("<DOCNO>") and line.endswith("</DOCNO>"):
                    docno = line.replace("<DOCNO>", "").replace("</DOCNO>", "")
                    current_doc["docno"] = docno

                # Start of text section
                elif line == "<TEXT>":
                    in_text_section = True

                # End of text section
                elif line == "</TEXT>":
                    in_text_section = False
                    current_doc["text"] = "\n".join(text_lines)

                # End of document
                elif line == "</DOC>":
                    # Check if this is the document we're looking for
                    if current_doc.get("docno") == doc_id:
                        # Extract URL (usually the first line of text)
                        text = current_doc.get("text", "")
                        lines = text.split("\n")
                        url = lines[0] if lines else "No URL found"

                        # Extract title (usually the second line)
                        title = lines[1] if len(lines) > 1 else "No title found"

                        # Body is the rest
                        body = (
                            "\n".join(lines[2:])
                            if len(lines) > 2
                            else "No body content"
                        )

                        return {
                            "docno": current_doc["docno"],
                            "url": url,
                            "title": title,
                            "body": body[:300] + "..." if len(body) > 300 else body,
                        }

                # Collect text content
                elif in_text_section:
                    text_lines.append(line)

    except FileNotFoundError:
        print(f"Collection file not found: {collection_file}")
        return None
    except Exception as e:
        print(f"Error reading collection file: {e}")
        return None

    return None


def find_multiple_docs(
    doc_ids,
    collection_file="/home/azureuser/.ir_datasets/msmarco-document/collection.tsv.gz",
):
    """Find URLs for multiple document IDs efficiently (single pass through file)."""

    doc_ids_set = set(doc_ids)  # For faster lookup
    found_docs = {}

    try:
        # Open the gzipped file
        with gzip.open(collection_file, "rt", encoding="utf-8") as f:
            current_doc = {}
            in_text_section = False
            text_lines = []

            for line in f:
                line = line.strip()

                # Start of a new document
                if line == "<DOC>":
                    current_doc = {}
                    in_text_section = False
                    text_lines = []

                # Document number
                elif line.startswith("<DOCNO>") and line.endswith("</DOCNO>"):
                    docno = line.replace("<DOCNO>", "").replace("</DOCNO>", "")
                    current_doc["docno"] = docno

                # Start of text section
                elif line == "<TEXT>":
                    in_text_section = True

                # End of text section
                elif line == "</TEXT>":
                    in_text_section = False
                    current_doc["text"] = "\n".join(text_lines)

                # End of document
                elif line == "</DOC>":
                    # Check if this is one of the documents we're looking for
                    docno = current_doc.get("docno")
                    if docno in doc_ids_set:
                        # Extract URL (usually the first line of text)
                        text = current_doc.get("text", "")
                        lines = text.split("\n")
                        url = lines[0] if lines else "No URL found"

                        # Extract title (usually the second line)
                        title = lines[1] if len(lines) > 1 else "No title found"

                        # Body is the rest
                        body = (
                            "\n".join(lines[2:])
                            if len(lines) > 2
                            else "No body content"
                        )

                        found_docs[docno] = {
                            "docno": docno,
                            "url": url,
                            "title": title,
                            "body": body[:300] + "..." if len(body) > 300 else body,
                        }

                        # Remove from set for efficiency
                        doc_ids_set.remove(docno)

                        # If we found all documents, we can stop
                        if not doc_ids_set:
                            break

                # Collect text content
                elif in_text_section:
                    text_lines.append(line)

    except FileNotFoundError:
        print(f"Collection file not found: {collection_file}")
        return {}
    except Exception as e:
        print(f"Error reading collection file: {e}")
        return {}

    return found_docs


def main():
    """Main function to check URLs by document IDs."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single document: python3 check_url.py <document_id>")
        print("  Multiple documents: python3 check_url.py <doc1> <doc2> <doc3> ...")
        print("Example: python3 check_url.py D2055699 D41271 D2487374")
        sys.exit(1)

    doc_ids = sys.argv[1:]

    if len(doc_ids) == 1:
        # Single document - use original function
        doc_id = doc_ids[0]
        print(f"Searching for document: {doc_id}")
        result = find_url_by_docid(doc_id)

        if result:
            print(f"\n=== Document {doc_id} ===")
            print(f"URL: {result['url']}")
            print(f"Title: {result['title']}")
            print(f"Body: {result['body']}")
        else:
            print(f"Document {doc_id} not found in collection")

    else:
        # Multiple documents - use efficient batch function
        print(f"Searching for {len(doc_ids)} documents...")
        results = find_multiple_docs(doc_ids)

        print(f"\nFound {len(results)} out of {len(doc_ids)} documents:\n")

        # Display results in the order they were requested
        for i, doc_id in enumerate(doc_ids, 1):
            print(f"=== Result #{i}: {doc_id} ===")
            if doc_id in results:
                result = results[doc_id]
                print(f"URL: {result['url']}")
                print(f"Title: {result['title']}")
                print(f"Body: {result['body']}")
            else:
                print("Document not found in collection")
            print()


if __name__ == "__main__":
    main()
