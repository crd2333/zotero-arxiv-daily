import arxiv
from datetime import datetime, timedelta

def test_query(query_str):
    print(f"Testing query: {query_str}")
    client = arxiv.Client(num_retries=3, delay_seconds=5)
    search = arxiv.Search(
        query=query_str,
        max_results=10,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    try:
        results = list(client.results(search))
        print(f"Found {len(results)} papers.")
        for r in results:
            print(f"  - Title: {r.title}")
            print(f"    Published: {r.published}")
            print(f"    Updated: {r.updated}")
    except Exception as e:
        print(f"Query failed: {e}")
    print("-" * 20)

if __name__ == "__main__":
    # Test cases based on your issue
    # 'cs.AI' and 'cs.CV'
    base_query = 'cat:cs.AI OR cat:cs.CV'

    # 1. Test submittedDate with YYYYMMDDHHMM format (strict) - What failed before
    test_query(f'({base_query}) AND submittedDate:[202512040000 TO 202512042359]')

    # 2. Test lastUpdatedDate with YYYYMMDDHHMM format - My proposed fix
    test_query(f'{base_query} AND lastUpdatedDate:[202512040000 TO 202512042359]')

    # # 3. Test a known valid historical date to ensure the format itself isn't broken
    # # Let's try 2024-01-01
    # test_query(f'{base_query} AND submittedDate:[202401010000 TO 202401012359]')

    # # 4. Test broader range (maybe just YYYYMMDD without time? API docs say YYYYMMDDHHMM is standard but let's check)
    # # Actually, API requires YYYYMMDDHHMM. Let's try expanding the window to 2 days to catch timezone issues.
    # test_query(f'{base_query} AND submittedDate:[202512041200 TO 202512061200]')
