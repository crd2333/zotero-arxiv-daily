from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
from datetime import datetime
import ast
import json

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _parse_categories_from_string(raw_categories: str) -> list[str]:
    stripped = raw_categories.strip()
    if not stripped:
        return []

    # Accept serialized list values from environment variables,
    # e.g. '["cs.AI","cs.CV"]' or "['cs.AI','cs.CV']".
    if stripped.startswith("[") and stripped.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                if isinstance(parsed, list):
                    return [str(category).strip() for category in parsed if str(category).strip()]
            except Exception:
                continue

        # Fallback for loosely formatted bracket strings.
        stripped = stripped[1:-1]

    if "," in stripped:
        return [item.strip().strip("\"'") for item in stripped.split(",") if item.strip()]

    if "+" in stripped:
        return [item.strip().strip("\"'") for item in stripped.split("+") if item.strip()]

    return [stripped.strip("\"'")]


def normalize_arxiv_categories(raw_categories: Any) -> list[str]:
    if raw_categories is None:
        raise ValueError("source.arxiv.category must be specified.")

    if isinstance(raw_categories, str):
        categories = _parse_categories_from_string(raw_categories)
    elif isinstance(raw_categories, (list, tuple)):
        categories = [str(category).strip() for category in raw_categories if str(category).strip()]
    else:
        raise TypeError(
            "source.arxiv.category must be a list of category strings, "
            "or a string such as '[\"cs.AI\",\"cs.CV\"]', 'cs.AI,cs.CV', or 'cs.AI+cs.CV'."
        )

    if len(categories) == 0:
        raise ValueError("source.arxiv.category must contain at least one category")

    return categories


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.categories = normalize_arxiv_categories(self.config.source.arxiv.category)

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.categories)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        query_date = self.config.source.arxiv.get("query_date", None)

        if query_date:
            return self._retrieve_raw_papers_for_date(client, include_cross_list, str(query_date))

        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def _retrieve_raw_papers_for_date(
        self,
        client: arxiv.Client,
        include_cross_list: bool,
        query_date: str,
    ) -> list[ArxivResult]:
        try:
            target_date = datetime.strptime(query_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("source.arxiv.query_date must use YYYY-MM-DD format") from exc

        categories = self.categories
        if len(categories) == 0:
            raise ValueError("source.arxiv.category must contain at least one category")

        category_query = " OR ".join([f"cat:{category}" for category in categories])
        if len(categories) > 1:
            category_query = f"({category_query})"

        date_str = target_date.strftime("%Y%m%d")
        search_query = f"{category_query} AND submittedDate:[{date_str}0000 TO {date_str}2359]"
        logger.info(f"Retrieving arXiv papers for date {query_date} with query: {search_query}")

        search = arxiv.Search(
            query=search_query,
            max_results=1000,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
        )
        raw_papers = list(client.results(search))

        if not include_cross_list:
            raw_papers = [
                paper
                for paper in raw_papers
                if getattr(paper, "primary_category", None) in categories
            ]

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
