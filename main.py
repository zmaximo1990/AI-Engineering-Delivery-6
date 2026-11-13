# Load env before the modules that read API keys from the environment.
from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
import logging

from langgraph.graph.state import CompiledStateGraph

from factory import LLMFactory
from graph import ask, compile_graph
from sample import DEMO_QUESTIONS
from schemas import LLMConfig, LLMProvider
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the research orchestrator. The default mode reads questions until exit or quit."
    )
    parser.add_argument(
        "-l",
        "--llm",
        default=settings.default_provider,
        choices=[provider.value for provider in LLMProvider],
        help="LLM provider (default: openai, or LLM_PROVIDER)",
    )
    parser.add_argument(
        "-d",
        "--demo",
        action="store_true",
        help="Run the sample questions, write the trace, and exit.",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Truncate trace.log before running.",
    )
    return parser.parse_args()


def _read_question() -> str:
    try:
        return input("Question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"


async def _run_demo(graph: CompiledStateGraph) -> None:
    print(
        f"recursion_limit={settings.recursion_limit}  "
        f"max_supervisor_steps={settings.max_supervisor_steps}",
        flush=True,
    )
    for label, question in DEMO_QUESTIONS:
        print(f"\n--- {label} ---", flush=True)
        print(f"Question: {question}", flush=True)
        turn = await ask(graph, question)
        print(turn, flush=True)
    logger.info("Trace appended to %s", settings.trace_log_path)
    logger.info("Finished")


async def _run_loop(graph: CompiledStateGraph) -> None:
    print(
        f"recursion_limit={settings.recursion_limit}  "
        f"max_supervisor_steps={settings.max_supervisor_steps}",
        flush=True,
    )
    print("Each question starts from a new state.", flush=True)
    print("Write a question and press Enter. exit or quit to terminate.\n", flush=True)
    while True:
        question = _read_question()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            turn = await ask(graph, question)
        except Exception as exc:
            logger.error("The orchestrator failed: %s", exc)
            continue
        print(turn, flush=True)
    logger.info("Finished")


async def main(args: argparse.Namespace) -> None:
    if args.clean:
        settings.trace_log_path.write_text("", encoding="utf-8")
        logger.info("Trace log reset at %s", settings.trace_log_path)

    config = LLMConfig(provider=LLMProvider(args.llm))
    try:
        LLMFactory.create_client(config)
    except ValueError as exc:
        logger.error("%s", exc)
        return

    logger.info("LLMConfig initialized (provider=%s)", config.provider.value)
    graph = compile_graph(config)
    if args.demo:
        await _run_demo(graph)
        return
    await _run_loop(graph)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
