# Travel RAG Assistant

This project is a travel planning assistant built with LangGraph and a Chroma DB-based RAG pipeline. It uses travel documents stored in a vector database to support retrieval-augmented travel planning.

## Overview

The current workflow is:

1. Ingest travel markdown documents into Chroma DB
2. Run the LangGraph-based travel planner
3. Generate a travel response using the retrieved context and LLM reasoning

## RAG Setup

RAG is implemented using Chroma DB.

- Travel documents are stored under the data/travel_docs folder
- The vector database is created and populated by ingest.py
- The retrieval layer is used for document-based context during the travel planning flow

## Important Run Order

Before running the main application, the vector database must be built first.

```bash
python ingest.py
python main.py
```

Run ingest.py again whenever the travel documents in the data/travel_docs folder are updated or changed.

## Features

- Natural-language travel request processing
- Structured travel data extraction using LLM output models
- Origin and destination IATA code identification
- Parallel flight and hotel information retrieval
- Flight data integration using Aviationstack
- Hotel discovery using Tavily Search
- AI-generated travel itinerary
- Final response aggregation using an LLM
- Shared workflow state using LangGraph
- LLM invocation tracking
- Chroma DB-powered RAG document retrieval
- Travel itinerary generation

## Workflow Architecture

The workflow follows this topology:

```text
START
  │
  ▼
Query Parser
  ├──────────────▶ Flight Agent ──────┐
  │                                   │
  └──────────────▶ Hotel Agent ───────┤
                                      ▼
                               Itinerary Agent
                                      │
                                      ▼
                                  Final Agent
                                      │
                                      ▼
                                     END
```

The Query Parser extracts structured travel details, including origin and destination information.

The workflow then branches into two independent retrieval nodes:

- The Flight Agent retrieves flight information using the Aviationstack API.
- The Hotel Agent discovers hotel information using Tavily Search.

Both branches update the shared graph state and converge at the Itinerary Agent.

The Itinerary Agent uses the combined flight and hotel context to generate a travel itinerary before the Final Agent prepares the consolidated response.

## Workflow

### 1. Query Parser

The Query Parser processes the user's natural-language travel request and extracts structured travel information.

Example input:

```text
"Plan a 5-day trip from Ahmedabad to ABC from September 11 including flights, hotels and sightseeing under ₹1 Lakh"
```

Example structured output:

```json
{
  "origin": "Ahmedabad",
  "origin_iata": "AMD",
  "destination": "ABC",
  "destination_iata": "ABC",
  "departure_date": "2026-07-11",
  "duration_days": 5
}
```

Optional information is not invented when it is missing from the user request.

### 2. Flight Agent

The Flight Agent uses the extracted travel details to retrieve flight information from the Aviationstack API.

### 3. Hotel Agent

The Hotel Agent uses Tavily Search to discover hotel information for the requested destination.

### 4. Itinerary Agent

The Itinerary Agent combines:

- The original user request
- Structured travel information
- Flight results
- Hotel results

The LLM generates a practical travel itinerary while distinguishing retrieved information from AI-generated recommendations.

### 5. Final Agent

The Final Agent aggregates the flight information, hotel information, and generated itinerary into a clear final travel response.

## Graph State

The workflow uses a shared LangGraph state.

```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    travel_request: TravelRequest
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
```

Each graph node reads the required state fields and returns only its state updates.

## Tech Stack

- Python
- LangGraph
- LangChain Core
- Groq
- Pydantic
- Aviationstack API
- Tavily Search
- Requests
- Chroma DB
- Hugging Face Embeddings

## API Keys

This project uses Groq for LLM inference, Aviationstack for flight information, and Tavily for hotel search.

### Groq API Key

1. Visit the Groq Cloud Console.
2. Create an account or sign in.
3. Open the API Keys section.
4. Create a new API key.
5. Add the key to your .env file.

### Aviationstack API Key

1. Visit Aviationstack.
2. Create an account or sign in.
3. Choose an available API plan.
4. Open your account dashboard and copy your API access key.
5. Add the key to your .env file.

### Tavily API Key

1. Visit Tavily.
2. Create an account or sign in.
3. Open the dashboard.
4. Generate or copy your API key.
5. Add the key to your .env file.

## Project Structure

```text
travel-rag-assistant/
├── ingest.py
├── main.py
├── requirements.txt
├── data/
│   └── travel_docs/
├── chroma_db/
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/buntynara/travel-rag-assistant.git
cd travel-rag-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a .env file with the required values:

```dotenv
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TRAVEL_DOC_PATH=./data/travel_docs
```

## Usage

Run the ingestion step first:

```bash
python ingest.py
```

Then run the main application:

```bash
python main.py
```

## Key Concepts Demonstrated

This project demonstrates:

- LangGraph graph-based LLM orchestration
- Shared state management across workflow nodes
- Structured LLM output using Pydantic models
- Natural-language to structured-data transformation
- Parallel workflow branches
- External REST API integration
- AI-oriented search integration
- Multi-source context aggregation
- LLM-based itinerary generation
- Separation of retrieval, reasoning, and response-generation stages

## Architecture Evolution

This project intentionally uses direct API and SDK integrations for external travel capabilities.

## Disclaimer

This project is intended for learning and architecture demonstration purposes.

Flight and hotel information depends on external API and search results. Generated itineraries are AI-assisted recommendations and should be verified before making travel decisions.
