# RAG Indexing Pipeline Documentation

This guide explains the step-by-step process of taking a PDF document and preparing it for a Retrieval-Augmented Generation (RAG) system, based on `src/indexing.js`.

## Overview
The goal of this pipeline is to take a PDF, extract its text, break the text down into manageable pieces, convert those pieces into numbers (embeddings), and save them in a database so they can be easily searched later.

## Libraries Needed

Here are the key libraries used and why we need them:

*   **`dotenv`**: Loads environment variables (like secret API keys) from a `.env` file so they aren't hardcoded in the code.
*   **`pdf-parse`**: Reads a PDF file and extracts all the raw text from it.
*   **`@langchain/core` & `@langchain/textsplitters`**: Provides tools to wrap our text into "Document" objects and intelligently break long text into smaller, overlapping chunks.
*   **`@google/genai`**: Connects to Google's Gemini AI to convert our text chunks into arrays of numbers (called embeddings).
*   **`@pinecone-database/pinecone` & `@langchain/pinecone`**: Connects to Pinecone, a specialized Vector Database designed to store and search through these arrays of numbers quickly.

---

## Step-by-Step Breakdown

### Step 1: Set up the Embeddings Interface
**What it does:** We configure a connection to Google's Gemini API and define how text should be converted into vectors (arrays of 3072 numbers).
**Why it's needed:** Computers don't understand words, they understand math. By converting text into vectors, we can mathematically compare how "similar" two pieces of text are. Similar concepts will have similar numbers.
*   `embedDocuments`: Used during this indexing phase to convert our PDF chunks.
*   `embedQuery`: Used later during the querying phase to convert the user's question into a vector for searching.

### Step 2: Load and Parse the PDF
**What it does:** We read the PDF file from the hard drive (`data/Node.pdf`) as raw bytes, and use `pdf-parse` to extract all the readable text. We then wrap this massive block of text into a LangChain `Document`.
**Why it's needed:** Before we can search or process the knowledge, we need to extract the raw text out of the proprietary PDF format. 

### Step 3: Split the Document into Chunks
**What it does:** We use a `RecursiveCharacterTextSplitter` to chop the massive block of PDF text into smaller chunks (e.g., 1000 characters each), with a slight overlap (e.g., 200 characters) between adjacent chunks. We also filter out any chunks that are empty or too small.
**Why it's needed:** 
1.  **Precision:** If we stored the whole PDF as one chunk, a search would just return the whole PDF, which isn't helpful. Smaller chunks mean search results are specific paragraphs or sections.
2.  **AI Limits:** LLMs (like ChatGPT or Gemini) have limits on how much text they can process at once (token limits). Breaking it down ensures we only send the most relevant pieces.
3.  **Overlap:** Overlapping chunks ensures we don't accidentally cut a sentence or important concept in half right at the boundary of a chunk.

### Step 4: Connect to Pinecone and Store Documents
**What it does:** We authenticate with the Pinecone Vector Database using our API key, select the specific "Index" (like a table in a database), and use a `PineconeStore` to upload everything. The store takes each text chunk, calls the Gemini embedding function from Step 1 to get the numbers, and saves both the text and the numbers to Pinecone.
**Why it's needed:** We need a permanent place to store all these vectors so that when a user asks a question, our system can instantly search the database for the closest matching vectors and retrieve the original text chunks. Pinecone is built specifically for this type of fast vector similarity search.
