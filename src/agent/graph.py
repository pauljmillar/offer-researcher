import asyncio
from typing import cast, Any, Literal
import json
from datetime import datetime
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import time
from functools import wraps
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log
)

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_sources, format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
    COMPARISON_PROMPT,
)

from anthropic import RateLimitError, APIError

# Load environment variables from .env
load_dotenv()

# Initialize Tavily client with API key from .env
tavily_async_client = AsyncTavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


class ComparisonOutput(BaseModel):
    """Structured output for the comparison step."""
    significant_changes: list[dict] = Field(
        default_factory=list,
        description="List of changes that require verification",
        # Define the expected structure of each change
        json_schema_extra={
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "old_value": {"type": "string"},
                    "new_value": {"type": "string"},
                    "requires_verification": {"type": "boolean"},
                    "verification_reason": {"type": "string"}
                }
            }
        }
    )
    verification_needed: bool = Field(
        default=False,
        description="Whether any changes need verification"
    )
    reasoning: str = Field(
        description="Explanation of why changes were or weren't flagged"
    )


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions with missing fields info
    query_instructions = QUERY_WRITER_PROMPT.format(
        card=state.card,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
        missing_fields=", ".join(state.missing_fields) if state.missing_fields else "all fields",
        reflection_reasoning=state.reflection_reasoning if state.reflection_reasoning else "Initial search"
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


# Define the retry decorator for API calls
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_exponential_jitter(initial=10, max=120),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def api_call_with_retry(func):
    """Decorator that implements exponential backoff for API calls."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Add base delay between all API calls
            await asyncio.sleep(3)  # 3 second base delay
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"API call failed: {str(e)}")
            raise
    return wrapper


# Apply the decorator to functions that make API calls
@api_call_with_retry
async def research_card(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process."""

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Track previously seen URLs to avoid duplicates across iterations
    previously_seen_urls = set()
    search_results = getattr(state, "search_results", [])
    if search_results:  # Only process if search_results exists and is not None
        previously_seen_urls = set(
            result["url"] 
            for results in search_results 
            for result in results
        )

    # Search tasks
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
                exclude_domains=list(previously_seen_urls)  # Exclude previously seen URLs
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    deduplicated_search_docs = deduplicate_sources(search_docs)
    source_str = format_sources(
        deduplicated_search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        card=state.card,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(p)
    state_update = {
        "completed_notes": [str(result.content)],
    }
    if configurable.include_search_results:
        state_update["search_results"] = deduplicated_search_docs

    return state_update


@api_call_with_retry
async def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = await structured_llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result}


@api_call_with_retry
async def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries."""
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        )
    )

    if result.is_satisfactory:
        return {
            "is_satisfactory": result.is_satisfactory,
            "reflection_reasoning": "Information is complete and satisfactory"
        }
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
            "reflection_reasoning": result.reasoning,
            "missing_fields": result.missing_fields
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_card"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_card"

    # If we've exceeded max steps, end even if not satisfactory
    return END


def clean_filename(name: str) -> str:
    """Clean card name for use in filenames."""
    # Convert to lowercase and remove escaped unicode characters
    cleaned = name.lower().encode('ascii', 'ignore').decode()
    
    # Remove common terms
    cleaned = cleaned.replace(' credit card', '').replace('credit card ', '')
    cleaned = cleaned.replace(' card', '').replace('card ', '')
    
    # Replace spaces and hyphens with underscores
    cleaned = cleaned.replace(' ', '_').replace('-', '_')
    
    # Remove any double underscores that might have been created
    cleaned = cleaned.replace('__', '_').strip('_')
    
    return cleaned


def save_card_snapshot(state: OverallState) -> dict[str, Any]:
    """Save the current card data to a JSON file."""
    logger.info(f"Saving snapshot for card: {state.card}")
    
    # Get standardized name from extracted info
    if state.info and isinstance(state.info, dict):
        standardized_name = clean_filename(state.info.get('card_name', '').strip())
    else:
        # Fallback to input name if extraction hasn't occurred
        standardized_name = clean_filename(state.card.strip())
    
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "card_name": state.card,
        "standardized_name": standardized_name,  # Include for reference
        "data": state.info,
        "sources": state.search_results if hasattr(state, "search_results") else []
    }
    
    # Ensure directories exist
    Path("snapshots").mkdir(exist_ok=True)
    
    # Save snapshot using standardized name
    filename = f"snapshots/{standardized_name}_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(filename, "w") as f:
        json.dump(snapshot, f, indent=2)
    
    logger.info(f"Saved snapshot to: {filename}")
    return {"snapshot_file": filename}


@api_call_with_retry
async def compare_with_previous(state: OverallState) -> dict[str, Any]:
    """Compare current results with previous snapshot."""
    logger.info(f"Comparing results for card: {state.card}")
    
    # Find most recent previous snapshot
    card_prefix = clean_filename(state.card)
    snapshots = sorted(Path("snapshots").glob(f"{card_prefix}_*.json"))
    
    if len(snapshots) <= 1:
        logger.info("No previous snapshot found for comparison")
        return {
            "changes_detected": False,
            "changes": [],
            "verification_needed": False,
            "comparison_steps_taken": state.comparison_steps_taken + 1
        }
    
    # Load previous snapshot
    logger.info(f"Comparing with previous snapshot: {snapshots[-2]}")
    with open(snapshots[-2]) as f:
        previous = json.load(f)
    
    # Only send relevant fields to the LLM
    relevant_fields = ['card_name', 'rewards_program_description', 'incentives', 'benefits']
    previous_data = {k: v for k, v in previous["data"].items() if k in relevant_fields}
    current_data = {k: v for k, v in state.info.items() if k in relevant_fields}
    
    system_prompt = COMPARISON_PROMPT.format(
        previous_data=json.dumps(previous_data, indent=2),
        current_data=json.dumps(current_data, indent=2)
    )
    
    # Use structured output for comparison
    structured_llm = claude_3_5_sonnet.with_structured_output(ComparisonOutput)
    result = cast(
        ComparisonOutput,
        await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Compare the two snapshots and identify significant changes."},
            ]
        ),
    )
    
    # Log comparison results
    if result.significant_changes:
        logger.info(f"Found {len(result.significant_changes)} significant changes")
        for change in result.significant_changes:
            logger.info(
                f"Change in {change['field']}: {change['old_value']} -> {change['new_value']}"
            )
    else:
        logger.info("No significant changes found")
    
    return {
        "changes_detected": bool(result.significant_changes),
        "changes": result.significant_changes,
        "verification_needed": result.verification_needed,
        "comparison_steps_taken": state.comparison_steps_taken + 1,
        "comparison_reasoning": result.reasoning
    }


def route_from_comparison(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "generate_queries"]:  # type: ignore
    """Route the graph based on the comparison output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If verification isn't needed or we've hit max steps, end
    if not state.verification_needed or state.comparison_steps_taken >= configurable.max_comparison_steps:
        return END

    # Continue with verification
    return "generate_queries"


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_card", research_card)
builder.add_node("reflection", reflection)
builder.add_node("save_snapshot", save_card_snapshot)
builder.add_node("compare_with_previous", compare_with_previous)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_card")
builder.add_edge("research_card", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges(
    "reflection",
    route_from_reflection
)
builder.add_edge("reflection", "save_snapshot")
builder.add_edge("save_snapshot", "compare_with_previous")
builder.add_conditional_edges(
    "compare_with_previous",
    route_from_comparison
)

# Compile
graph = builder.compile()
