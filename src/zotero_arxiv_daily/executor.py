import os
import time
from types import SimpleNamespace
from typing import Any
from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
DEFAULT_LOCAL_LLM_MAX_SECONDS = 1800


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def parse_use_llm_api_env(raw_value: str | None) -> bool | None:
    """Compatibility parser for USE_LLM_API.

    In this fork, USE_LLM_API=1 means "use local LLM", while 0 means "use API".
    """
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if normalized == "":
        return None
    if normalized in TRUE_ENV_VALUES:
        return True
    if normalized in FALSE_ENV_VALUES:
        return False

    logger.warning(
        f"Invalid USE_LLM_API value '{raw_value}'. Expected one of {sorted(TRUE_ENV_VALUES | FALSE_ENV_VALUES)}."
    )
    return None


def parse_non_negative_int_env(raw_value: str | None, env_name: str) -> int | None:
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if normalized == "":
        return None

    try:
        value = int(normalized)
    except ValueError:
        logger.warning(f"Invalid {env_name} value '{raw_value}'. Expected a non-negative integer.")
        return None

    if value < 0:
        logger.warning(f"Invalid {env_name} value '{raw_value}'. Expected a non-negative integer.")
        return None

    return value


def should_use_local_llm(config: DictConfig) -> bool:
    env_value = parse_use_llm_api_env(os.getenv("USE_LLM_API"))
    if env_value is not None:
        return env_value

    # Config fallback for local runs without explicit USE_LLM_API env.
    if "use_local" in config.llm and config.llm.use_local is not None:
        return bool(config.llm.use_local)
    return False


def resolve_local_llm_max_seconds(config: DictConfig) -> int:
    env_value = parse_non_negative_int_env(os.getenv("LOCAL_LLM_MAX_SECONDS"), "LOCAL_LLM_MAX_SECONDS")
    if env_value is not None:
        return env_value

    if (
        "local" in config.llm
        and "max_generation_seconds" in config.llm.local
        and config.llm.local.max_generation_seconds is not None
    ):
        try:
            return int(config.llm.local.max_generation_seconds)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid config.llm.local.max_generation_seconds value "
                f"'{config.llm.local.max_generation_seconds}'. Use default {DEFAULT_LOCAL_LLM_MAX_SECONDS}s."
            )

    return DEFAULT_LOCAL_LLM_MAX_SECONDS


class LocalLLMClient:
    """OpenAI-compatible wrapper around local llama.cpp chat completion."""

    def __init__(self, llm_config: DictConfig):
        self.llm_config = llm_config
        self._llm = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )

    def _get_local_setting(self, key: str, default: Any) -> Any:
        if "local" not in self.llm_config:
            return default
        local_cfg = self.llm_config.local
        if key not in local_cfg or local_cfg[key] is None:
            return default
        return local_cfg[key]

    def _load_model(self):
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM backend requires llama-cpp-python. "
                "Install it or set USE_LLM_API=0 to use API backend."
            ) from exc

        model = self._get_local_setting("model", "Qwen/Qwen2.5-3B-Instruct-GGUF")
        filename = self._get_local_setting("filename", "qwen2.5-3b-instruct-q4_k_m.gguf")
        n_ctx = int(self._get_local_setting("n_ctx", 8192))
        n_threads = int(self._get_local_setting("n_threads", 4))

        logger.info(f"Loading local LLM model {model}:{filename}")
        self._llm = Llama.from_pretrained(
            repo_id=model,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        return self._llm

    @staticmethod
    def _normalize_response(response: Any):
        if hasattr(response, "choices"):
            return response

        if isinstance(response, dict):
            choices = []
            for choice in response.get("choices", []):
                message = choice.get("message", {})
                choices.append(
                    SimpleNamespace(
                        message=SimpleNamespace(content=message.get("content")),
                        finish_reason=choice.get("finish_reason"),
                        index=choice.get("index", 0),
                    )
                )
            return SimpleNamespace(choices=choices)

        raise TypeError("Unexpected local LLM response format")

    def _create_chat_completion(self, *args, **kwargs):
        llm = self._load_model()
        kwargs = dict(kwargs)
        # llama.cpp backend ignores remote model name and uses loaded GGUF model.
        kwargs.pop("model", None)
        response = llm.create_chat_completion_openai_v1(*args, **kwargs)
        return self._normalize_response(response)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.use_local_llm = should_use_local_llm(config)
        self.local_llm_max_seconds = resolve_local_llm_max_seconds(config)
        if self.use_local_llm:
            logger.info("USE_LLM_API enabled local LLM mode (1=local, 0=API).")
            if self.local_llm_max_seconds > 0:
                logger.info(
                    f"Local LLM generation time budget: {self.local_llm_max_seconds}s per run."
                )
            else:
                logger.info("Local LLM generation time budget is disabled (max seconds = 0).")
            self.openai_client = LocalLLMClient(config.llm)
        else:
            self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]

    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus


    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            local_llm_deadline = None
            if self.use_local_llm and self.local_llm_max_seconds > 0:
                local_llm_deadline = time.monotonic() + self.local_llm_max_seconds

            for paper_index, p in enumerate(tqdm(reranked_papers), start=1):
                if local_llm_deadline is not None and time.monotonic() >= local_llm_deadline:
                    logger.warning(
                        "Reached local LLM generation time budget "
                        f"({self.local_llm_max_seconds}s). "
                        f"Stop after {paper_index - 1}/{len(reranked_papers)} papers."
                    )
                    break
                p.generate_tldr(self.openai_client, self.config.llm)

                if local_llm_deadline is not None and time.monotonic() >= local_llm_deadline:
                    logger.warning(
                        "Reached local LLM generation time budget after TLDR generation. "
                        f"Stop after {paper_index}/{len(reranked_papers)} papers."
                    )
                    break

                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
