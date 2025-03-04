EXTRACTION_PROMPT = """Your task is to take notes gathered from credit card research and extract them into the following schema.

<schema>
{info}
</schema>

Here are all the notes from research:

<web_research_notes>
{notes}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific credit card information.

Here is the credit card you are researching: {card}

Generate at most {max_search_queries} search queries that will help gather the following information:

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Your query should:
1. Focus on finding current, accurate credit card details and offers
2. Target official bank websites, credit card comparison sites, and reliable financial sources
3. Prioritize finding information that matches the schema requirements
4. Include the specific card name, issuer, and relevant terms
5. Be specific enough to avoid irrelevant credit card results
6. Look for both benefits and fee information

Create focused queries that will maximize the chances of finding accurate, up-to-date credit card information."""

INFO_PROMPT = """You are doing web research on a credit card, {card}. 

The following schema shows the type of information we're interested in:

<schema>
{info}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about the credit card, focusing on topics relevant to our interests.

<Website contents>
{content}
</Website contents>

Here are any additional notes from the user:
<user_notes>
{user_notes}
</user_notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema (fees, rewards, terms, etc.)
3. Include specific numbers, rates, and terms when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear
6. Pay special attention to any terms, conditions, or limitations
7. Distinguish between introductory and regular rates/fees

At the end of your notes, please include:
8. A "Sources:" section listing all websites referenced in the research

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

REFLECTION_PROMPT = """You are a financial analyst tasked with reviewing the quality and completeness of extracted credit card information.

Compare the extracted information with the required schema:

<Schema>
{schema}
</Schema>

Here is the extracted information:
<extracted_info>
{info}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:

Basic Card Information:
1. Are the required fields (card name, issuer, network) present and accurate?
2. Is the card industry properly identified?

Fees and APRs:
3. Are both introductory and regular fees clearly distinguished?
4. Are APR ranges properly specified with both low and high values?
5. Are introductory APR durations specified in months?
6. Are all fee types (annual, balance transfer, late fees) documented?

Rewards and Benefits:
7. Is the rewards program fully described with type, earning rates, and categories?
8. Are reward expiration policies and triggers clearly explained?
9. Are redemption options and limitations documented?
10. Are card benefits listed with clear descriptions?
11. Are alternative payment plans and contactless features specified?

Incentives:
12. Are sign-up bonuses or welcome offers properly documented?
13. Do incentives include clear descriptions, types, and values?

Quality Checks:
14. Are there any fields with placeholder values or "unknown" markers?
15. Is the information current and consistent?
16. Are there any contradictions in the data?

Provide specific recommendations for any missing or unclear information."""
