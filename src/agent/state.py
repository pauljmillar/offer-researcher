from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator
from pydantic import BaseModel, Field, ConfigDict


DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CreditCardInfo",
    "description": "Comprehensive credit card information including attributes, fees, APRs, rewards, and benefits",
    "type": "object",
    "properties": {
        # Basic Card Info
        "card_name": {
            "type": "string",
            "description": "The name of the credit card."
        },
        "card_issuer": {
            "type": "string",
            "description": "The bank or financial institution that extends the line of credit."
        },
        "card_network": {
            "type": "string",
            "description": "The company that helps process each of your transactions.",
            "enum": ["American Express", "Discover", "Mastercard", "Visa"]
        },
        "card_industry": {
            "type": "string",
            "description": "The industry that the credit card specializes in."
        },
        # Fees
        "has_annual_fee": {
            "type": "string",
            "description": "Whether there is an annual fee for the card.",
            "enum": ["Y", "N"]
        },
        "intro_annual_fee": {
            "type": "number",
            "description": "Annual fee amount for the during the introductory period."
        },
        "regular_annual_fee": {
            "type": "number",
            "description": "Annual fee amount ongoing, after the introductory period."
        },
        "intro_balance_transfer_fee": {
            "type": "number",
            "description": "Balance transfer fee amount during the introductory period."
        },
        "regular_balance_transfer_fee": {
            "type": "number",
            "description": "Balance transfer fee amount after the introductory period."
        },
        "first_late_fee": {
            "type": "number",
            "description": "Late fee charged for the FIRST infraction."
        },
        "ongoing_late_fee": {
            "type": "number",
            "description": "Late fee charged AFTER the first infraction."
        },
        # APRs
        "regular_apr_low": {
            "type": "string",
            "description": "For the regular APR, what is the LOWEST end of the range."
        },
        "regular_apr_high": {
            "type": "string",
            "description": "For the regular APR, what is the HIGHEST end of the range."
        },
        "intro_purchase_apr": {
            "type": "number",
            "description": "APR on purchases during the introductory period."
        },
        "intro_purchase_apr_duration": {
            "type": "number",
            "description": "In months, the duration of the introductory APR period for purchases."
        },
        "intro_balance_transfer_apr": {
            "type": "number",
            "description": "APR on balance transfers during the introductory period."
        },
        "intro_balance_transfer_apr_duration": {
            "type": "number",
            "description": "In months, the duration of the introductory APR period for balance transfers."
        },
        # Rewards
        "rewards_type": {
            "type": "string",
            "description": "The type of rewards offered by this credit card.",
            "enum": ["Airline Miles", "Hotel Points", "Other Points", "Cash Back", "Other"]
        },
        "rewards_program_description": {
            "type": "string",
            "description": "A text description of the rewards program. This should explain the categories and earn rates, etc."
        },
        "rewards_expiration": {
            "type": "string",
            "description": "Do the rewards expire?",
            "enum": ["Y", "N"]
        },
        "rewards_expiration_detail": {
            "type": "string",
            "description": "An explanation of the rewards expiration policy."
        },
        "rewards_trigger": {
            "type": "string",
            "description": "Whether there is a rewards trigger.",
            "enum": ["Y", "N"]
        },
        "rewards_trigger_detail": {
            "type": "string",
            "description": "An explanation of the rewards trigger."
        },
        "rewards_redemption_options": {
            "type": "string",
            "description": "Information about the process or limitations relating to the rewards redemption."
        },
        # Benefits
        "alternative_payment_plans": {
            "type": "string",
            "description": "List alternative payment plans offered by the card, if any, including Amex Pay It, Plan It; Citi Flex Pay/Plan, My Chase Plan/Loan."
        },
        "contactless_payment": {
            "type": "string",
            "description": "Does the card offer contactless payment, also referred to as tap-to-pay?",
            "enum": ["Y", "N"]
        },
        "benefits": {
            "type": "array",
            "items": {"$ref": "#/$defs/Benefits"}
        },
        "incentives": {
            "type": "array",
            "items": {"$ref": "#/$defs/Incentives"}
        }
    },
    "$defs": {
        "Incentives": {
            "type": "object",
            "description": "Incentives offered to encourage a new card holder to apply",
            "properties": {
                "incentive_description": {
                    "type": "string",
                    "description": "Text description of the incentive offer."
                },
                "incentive_type": {
                    "type": "string",
                    "description": "The categorization of the incentive.",
                    "enum": ["Rewards", "Miles", "Cash", "Points", "Low Introductory Rate"]
                },
                "incentive_value": {
                    "type": "number",
                    "description": "The amount of the incentive offer."
                }
            }
        },
        "Benefits": {
            "type": "object",
            "description": "Benefits offered to card holders",
            "properties": {
                "benefit_description": {
                    "type": "string",
                    "description": "Text description of the benefit."
                }
            }
        }
    },
    "required": ["card_name", "card_issuer", "card_network"]
}


class BaseStateModel(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        extra='forbid',
        frozen=False,  # Allow field updates
        validate_assignment=True,
        revalidate_instances='always'
    )

    # Define common fields that all states will inherit
    card: str
    user_notes: str = Field(default="")
    extraction_schema: dict[str, Any] = Field(default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA)
    search_queries: Annotated[list[str], operator.add] = Field(default_factory=list)
    search_results: list[dict] = Field(default_factory=list)
    completed_notes: Annotated[list[str], operator.add] = Field(default_factory=list)
    info: Optional[dict[str, Any]] = Field(default=None)
    is_satisfactory: bool = Field(default=False)
    reflection_steps_taken: int = Field(default=0)
    reflection_reasoning: str = Field(default="")
    missing_fields: list[str] = Field(default_factory=list)
    changes: list[dict] = Field(default_factory=list)
    verification_needed: bool = Field(default=False)
    snapshot_file: Optional[str] = Field(default=None)
    comparison_steps_taken: int = Field(default=0)

# Inherit all fields from BaseStateModel
class InputState(BaseStateModel):
    """Input state defines the interface between the graph and the user."""
    pass

class OverallState(BaseStateModel):
    """The overall state of the graph."""
    pass

class OutputState(BaseStateModel):
    """The output state of the graph."""
    pass
