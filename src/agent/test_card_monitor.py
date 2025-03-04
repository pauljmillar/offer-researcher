# test_card_monitor.py
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from agent.graph import graph
from agent.state import DEFAULT_EXTRACTION_SCHEMA, InputState
from agent.config import CardConfig, load_monitoring_settings
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_card_with_langsmith(card_name: str):
    """Run single card with LangSmith tracking."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    return await run_single_card(card_name)

async def run_card_batch_mode(card_name: str):
    """Run single card in batch mode (file-based)."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return await run_single_card(card_name)

async def run_single_card(card_name: str):
    """Core function to run the graph for a single card."""
    # Ensure snapshots directory exists
    Path("snapshots").mkdir(exist_ok=True)
    
    # Create input state with all required fields
    input_state = InputState(
        card=card_name,
        user_notes="",
        extraction_schema=DEFAULT_EXTRACTION_SCHEMA,
        search_queries=[],
        search_results=[],
        completed_notes=[],
        info=None,
        is_satisfactory=False,
        reflection_steps_taken=0,
        reflection_reasoning="",
        missing_fields=[],
        changes=[],
        verification_needed=False,
        snapshot_file=None,
        comparison_steps_taken=0
    )
    
    logger.info(f"Starting research for card: {input_state.card}")
    return await graph.ainvoke(input_state)

async def process_cards(cards: List[CardConfig], parallel: bool = False) -> List[dict]:
    """Process multiple cards either sequentially or in parallel."""
    results = []
    
    if parallel:
        # Process cards in parallel
        tasks = [run_single_card(card.name) for card in cards if card.active]
        results = await asyncio.gather(*tasks)
    else:
        # Process cards sequentially
        for card in cards:
            if card.active:
                result = await run_single_card(card.name)
                results.append(result)
                # Add delay between cards to manage rate limits
                await asyncio.sleep(5)
    
    return results

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_monitoring_settings()
    
    # Process all active cards
    results = asyncio.run(
        process_cards(
            config.monitoring.cards,
            parallel=os.getenv("PARALLEL", "false").lower() == "true"
        )
    )
    
    # Print summary
    print("\n=== Monitoring Summary ===")
    for card, result in zip(config.monitoring.cards, results):
        if not card.active:
            continue
        print(f"\nCard: {card.name}")
        print(f"Changes detected: {bool(result.get('changes', []))}")
        if result.get('changes'):
            print("\nChanges:")
            for change in result['changes']:
                print(f"- {change['field']}: {change['old_value']} -> {change['new_value']}")