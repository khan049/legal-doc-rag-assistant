import re

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
)


# --------------------------------------------------
# Models / Clients
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)

llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def search(query, limit=12):

    vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
    )

    return [
        point
        for point in results.points
        if point.score >= 0.35
    ]


# --------------------------------------------------
# Build context with source IDs
# --------------------------------------------------

def build_context(points):

    context_parts = []

    citations = []

    for index, point in enumerate(points, start=1):

        payload = point.payload

        citation = {
            "id": index,
            "document": payload["document"],
            "page": payload["page"],
            "text": payload["text"],
            "score": point.score,
        }

        citations.append(citation)

        context_parts.append(
            f"""
[SOURCE {index}]
Document: {payload["document"]}
Page: {payload["page"]}
Text:
{payload["text"]}
"""
        )

    return "\n".join(context_parts), citations


# --------------------------------------------------
# Ask LLM
# --------------------------------------------------

def ask_llm(question, context):

    prompt = f"""
You are a strict document question-answering system.

Your ONLY source of information is the SOURCES below.

IMPORTANT:

- Do NOT use your general knowledge.
- Do NOT guess.
- Do NOT combine unrelated passages just because they contain
  a word from the question.
- A source must directly provide information that answers the question.
- Do NOT create a "policy", rule, conclusion, or definition unless
  the supplied source explicitly provides it.
- If the sources only mention the question's keywords but do not
  actually answer the question, the answer is NOT available.
- When in doubt, return NOT AVAILABLE.

For example, if the question is:
"What is the leave policy?"

and the sources only discuss:
"leaves taken by a public employee"
or
"rights in respect of leave of absence"

that does NOT establish a general leave policy.
Therefore return NOT AVAILABLE.

If the answer is available, answer using only the exact facts
supported by the sources.

Return EXACTLY:

ANSWER: <answer>
CITATIONS: <source numbers>

OR, if the answer is not supported:

ANSWER: Information is not available in the supplied documents.
CITATIONS: NONE

Do not include any explanation outside this format.

SOURCES:

{context}

QUESTION:

{question}
"""

    try:
        response = llm.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
            error_message = str(e)

            if "429" in error_message:
                return (
                    "ERROR: OpenRouter free model is temporarily busy. "
                    "Please try the question again."
                )

            return f"ERROR: OpenRouter request failed: {e}"

# --------------------------------------------------
# Parse LLM response
# --------------------------------------------------

def parse_response(response):

    if response.startswith("ERROR:"):
        return response, []

    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=\nCITATIONS:|\Z)",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    citation_match = re.search(
        r"CITATIONS:\s*(.*)",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if not answer_match:
        return (
            "Information is not available in the supplied documents.",
            [],
        )

    answer = answer_match.group(1).strip()

    # Remove accidental citation text if model puts it inside answer
    answer = re.sub(
        r"\s*\(?CITATIONS?:.*$",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    if not citation_match:
        return answer, []

    citation_text = citation_match.group(1).strip()

    if citation_text.upper() == "NONE":
        return answer, []

    source_ids = []

    for number in re.findall(r"\d+", citation_text):

        source_id = int(number)

        if source_id <= 12 and source_id not in source_ids:
            source_ids.append(source_id)

    return answer, source_ids


# --------------------------------------------------
# Print citations
# --------------------------------------------------

def print_sources(citations, source_ids):

    selected = []

    for source_id in source_ids:

        for citation in citations:

            if citation["id"] == source_id:
                selected.append(citation)
                break

    # Remove duplicate document/page combinations
    shown = set()

    for citation in selected:

        key = (
            citation["document"],
            citation["page"],
        )

        if key in shown:
            continue

        shown.add(key)

        print("-" * 60)

        print(f'Document : {citation["document"]}')
        print(f'Page     : {citation["page"]}')
        print("Retrieved Text:")

        print(citation["text"])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("RAG Legal Document Assistant")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        # Optional terminal clear command
        if question.lower() == "cls":
            import os
            os.system("cls" if os.name == "nt" else "clear")
            continue

        try:

            points = search(question)

            if not points:

                print(
                    "\nAnswer\n\n"
                    "Information is not available in the supplied documents.\n"
                )

                continue

            context, citations = build_context(points)

            response = ask_llm(
                question,
                context,
            )

            answer, source_ids = parse_response(response)

            print("\nAnswer\n")
            print(answer)

            # Don't show sources for unavailable answers
            if (
                answer.strip()
                == "Information is not available in the supplied documents."
            ):
                print()
                continue

            if source_ids:

                print("\nSources\n")

                print_sources(
                    citations,
                    source_ids,
                )

            else:

                print(
                    "\nInformation could not be reliably "
                    "supported by the retrieved documents."
                )

            print()

        except Exception as e:

            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()