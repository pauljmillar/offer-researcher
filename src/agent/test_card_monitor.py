# test_card_monitor.py
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from agent.graph import graph
from agent.state import DEFAULT_EXTRACTION_SCHEMA, InputState

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

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if running in batch mode
    BATCH_MODE = os.getenv("BATCH_MODE", "false").lower() == "true"
    
    # Example card
    card_name = "Chase Sapphire Preferred"
    
    # Run in appropriate mode
    result = asyncio.run(
        run_card_batch_mode(card_name) if BATCH_MODE 
        else run_card_with_langsmith(card_name)
    )
    
    # Print results
    print("\n=== Results ===")
    print(f"Card: {card_name}")
    print(f"Is satisfactory: {getattr(result, 'is_satisfactory', False)}")
    print(f"Reflection steps taken: {getattr(result, 'reflection_steps_taken', 0)}")
    print(f"\nReflection reasoning: {getattr(result, 'reflection_reasoning', '')}")
    
    missing_fields = getattr(result, 'missing_fields', [])
    if missing_fields:
        print("\nMissing or incomplete fields:")
        for field in missing_fields:
            print(f"- {field}")
    
    snapshot_file = getattr(result, 'snapshot_file', None)
    if snapshot_file:
        print(f"\nSnapshot saved to: {snapshot_file}")
    
    changes = getattr(result, 'changes', [])
    if changes:
        print("\nChanges detected:")
        for change in changes:
            print(f"\nField: {change['field']}")
            print(f"  Old value: {change['old_value']}")
            print(f"  New value: {change['new_value']}")
            print(f"  Verified: {change['verified']}")
    else:
        print("\nNo changes detected from previous snapshot")