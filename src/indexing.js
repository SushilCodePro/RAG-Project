// ============================================================================
// indexing.js — The INDEXING pipeline for RAG
//
// PURPOSE: Takes a PDF file → extracts text → splits into chunks → 
//          converts chunks to number vectors (embeddings) → stores in Pinecone
//
// FLOW DIAGRAM:
//   PDF File 
//     → [pdf-parse] Read & extract text page by page
//     → [Document] Wrap each page's text into a LangChain Document object
//     → [RecursiveCharacterTextSplitter] Break big pages into small chunks
//     → [GoogleGenAI] Convert each chunk text into a 3072-dimension number array
//     → [PineconeStore] Upload those number arrays + text to Pinecone database
//
// WHEN TO RUN: Only once (or when your PDF data changes). 
//              This populates your Pinecone index so query.js can search it.
// ============================================================================

// ─── Step 0: Load environment variables ─────────────────────────────────────
// dotenv reads your .env file and puts values into process.env
// e.g., process.env.GEMINI_API_KEY, process.env.PINECONE_API_KEY
import * as dotenv from 'dotenv';
dotenv.config();
// After this line: process.env.GEMINI_API_KEY = "your-key-from-.env-file"

// ─── NEW IMPORTS (following latest official LangChain docs) ─────────────────

// readFileSync: Node.js built-in function that reads a file from disk
// Takes: file path (string)  →  Returns: file contents as Buffer (raw bytes)
import { readFileSync } from 'node:fs';

// Document: LangChain's standard wrapper for a piece of text + its metadata
// Takes: { pageContent: string, metadata: object }  →  Returns: Document object
// Think of it as a labeled container: the text + info about where it came from
import { Document } from '@langchain/core/documents';

// pdf: Parses a PDF buffer and extracts text, page count, metadata
// Takes: Buffer (raw PDF bytes)  →  Returns: { text, numpages, info }
// This REPLACES the old PDFLoader from @langchain/community
import pdf from 'pdf-parse';

// RecursiveCharacterTextSplitter: Breaks long text into smaller chunks
// WHY? LLMs have token limits, and smaller chunks = more precise search results
// Takes: Document[]  →  Returns: Document[] (more documents, each smaller)
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

// Pinecone: The client SDK to connect to Pinecone vector database
// Takes: { apiKey }  →  Returns: Pinecone client with .Index() method
import { Pinecone } from '@pinecone-database/pinecone';

// PineconeStore: LangChain's wrapper around Pinecone that handles
// embedding + storing + searching all in one place
// Takes: (embeddings, { pineconeIndex })  →  Returns: a vector store you can query
import { PineconeStore } from '@langchain/pinecone';

// GoogleGenAI: Google's AI SDK for calling Gemini models (embeddings + chat)
// Takes: { apiKey }  →  Returns: client with .models.embedContent() method
import { GoogleGenAI } from '@google/genai';

// ════════════════════════════════════════════════════════════════════════════
// OLD IMPORTS (commented out — these were the previous approach)
// ════════════════════════════════════════════════════════════════════════════
// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// 
// PDFLoader was a convenience wrapper that:
//   1. Read the PDF file internally
//   2. Extracted text using pdf-parse under the hood
//   3. Automatically created Document[] objects
//
// WHY REPLACED? The official LangChain docs now recommend using pdf-parse 
// directly because:
//   - Fewer dependencies (no need for @langchain/community)
//   - More control over how text is extracted and metadata is set
//   - PDFLoader was just a thin wrapper anyway


// ─── Step 1: Set up the Embeddings interface ────────────────────────────────
// 
// WHAT ARE EMBEDDINGS?
// Embeddings convert text → array of numbers (vector), e.g.:
//   "hello world" → [0.12, -0.45, 0.78, ... ] (3072 numbers)
//
// WHY? Computers can't understand text directly. By converting to numbers,
// we can mathematically compare how "similar" two pieces of text are.
// Similar text → similar number arrays → close together in vector space.
//
// This wrapper object gives PineconeStore two functions it needs:
//   - embedDocuments(): for converting document chunks (during indexing)
//   - embedQuery(): for converting search questions (during querying)
// ────────────────────────────────────────────────────────────────────────────

// Create a GoogleGenAI client — this connects to Google's Gemini API
// Takes: { apiKey: string }
// Returns: client object with methods like .models.embedContent()
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// This object follows LangChain's Embeddings interface
// PineconeStore expects an object with embedDocuments() and embedQuery()
const embeddings = {

  // embedDocuments: Called during INDEXING to convert document chunks to vectors
  // Takes: texts (string[]) — an array of chunk texts
  // Returns: number[][] — array of vectors, one per text
  // Example: ["chunk1 text", "chunk2 text"] → [[0.1, 0.2, ...], [0.3, 0.4, ...]]
  embedDocuments: async (texts) => {
    const vectors = [];
    for (const text of texts) {
      // genAI.models.embedContent() sends text to Google's API and gets back numbers
      // Takes: { model, contents, config }
      // Returns: { embeddings: [{ values: number[] }] }
      const res = await genAI.models.embedContent({
        model: 'gemini-embedding-001',   // The embedding model to use
        contents: text.trim(),            // The text to convert (trimmed of whitespace)
        config: { 
          taskType: 'RETRIEVAL_DOCUMENT'  // Tells Google: "this is a document to be searched"
        },
      });
      // res.embeddings[0].values = the actual number array (3072 dimensions)
      vectors.push(res.embeddings[0].values);
    }
    return vectors; // Returns: array of number arrays
  },

  // embedQuery: Called during QUERYING to convert the user's question to a vector
  // Takes: text (string) — the user's search question
  // Returns: number[] — a single vector
  // Example: "What is Node.js?" → [0.5, 0.1, -0.3, ...]
  embedQuery: async (text) => {
    const res = await genAI.models.embedContent({
      model: 'gemini-embedding-001',
      contents: text.trim(),
      config: { 
        taskType: 'RETRIEVAL_QUERY'  // Tells Google: "this is a search query"
        // NOTE: RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT — Google optimizes the 
        // vectors differently depending on whether it's a question or a document
      },
    });
    return res.embeddings[0].values; // Returns: single number array
  },
};


// ─── Step 2: The main indexing function ─────────────────────────────────────
async function indexing() {

  // ═══════════════════════════════════════════════════════════════════════════
  // STEP 2a: Load PDF and create Documents (NEW approach from official docs)
  // ═══════════════════════════════════════════════════════════════════════════
  
  const PDF_PATH = 'data/Node.pdf';
  console.log(`📄 Loading PDF from "${PDF_PATH}"...`);

  // readFileSync() reads the entire PDF file as raw bytes (Buffer)
  // Takes: file path (string)
  // Returns: Buffer — raw binary data of the PDF file
  const pdfBuffer = readFileSync(PDF_PATH);

  // pdf() parses the raw PDF bytes and extracts all text + metadata
  // Takes: Buffer (raw PDF bytes)
  // Returns: { text: string, numpages: number, info: object }
  //   - text: ALL text from the entire PDF combined
  //   - numpages: how many pages the PDF has
  //   - info: PDF metadata (title, author, etc.)
  const pdfData = await pdf(pdfBuffer);
  console.log(`✅ PDF loaded. Pages: ${pdfData.numpages}`);

  // Create a LangChain Document from the extracted text
  // Document is just a container: { pageContent: string, metadata: object }
  //   - pageContent: the actual text content
  //   - metadata: any extra info (source file, page number, etc.)
  //
  // NOTE: pdf-parse gives us all text combined. We wrap it in a Document
  // so that RecursiveCharacterTextSplitter can process it.
  const rawDocs = [
    new Document({
      pageContent: pdfData.text,               // The full text of the PDF
      metadata: { source: PDF_PATH },          // Where this text came from
    }),
  ];
  console.log(`📄 Created ${rawDocs.length} Document(s) from PDF`);

  // ════════════════════════════════════════════════════════════════════════
  // OLD PDF LOADING (commented out):
  // ════════════════════════════════════════════════════════════════════════
  // const pdfLoader = new PDFLoader(PDF_PATH);
  // const rawDocs = await pdfLoader.load();
  //
  // PDFLoader did steps 2a automatically:
  //   1. Read the file (readFileSync internally)
  //   2. Parse PDF (used pdf-parse internally)  
  //   3. Created Document[] (one per page, with page numbers in metadata)
  //
  // The new approach gives us more control but requires more code.
  // ════════════════════════════════════════════════════════════════════════


  // ═══════════════════════════════════════════════════════════════════════════
  // STEP 2b: Split documents into smaller chunks
  // ═══════════════════════════════════════════════════════════════════════════
  
  // WHY SPLIT? 
  //   - A 100-page PDF as one chunk → too big, search results are vague
  //   - 1000-character chunks → small enough for precise search, big enough for context
  //
  // RecursiveCharacterTextSplitter tries to split at natural boundaries:
  //   First tries: \n\n (paragraphs) → \n (lines) → " " (words) → "" (characters)
  //
  // Takes: { chunkSize, chunkOverlap }
  //   - chunkSize: max characters per chunk (1000 chars ≈ 250 words)
  //   - chunkOverlap: characters shared between adjacent chunks (prevents cutting sentences)
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,     // Each chunk will be at most 1000 characters
    chunkOverlap: 200,   // Adjacent chunks share 200 characters (so no info is lost at boundaries)
  });

  // splitDocuments() takes Document[] and returns Document[] (but more of them, smaller)
  // Takes: Document[] (big documents)
  // Returns: Document[] (many small documents, each ≤ 1000 chars)
  // Example: 1 document of 10,000 chars → ~12 documents of ~1000 chars each
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

  // Filter out empty or too-small chunks (chunks with < 50 chars are usually junk)
  // This prevents embedding errors from empty/whitespace-only chunks
  const cleanDocs = chunkedDocs.filter(doc => doc.pageContent.trim().length > 50);
  console.log(`✂️  Chunks: ${chunkedDocs.length} → after filtering empty: ${cleanDocs.length}`);


  // ═══════════════════════════════════════════════════════════════════════════
  // STEP 2c: Connect to Pinecone and store documents (NEW approach)
  // ═══════════════════════════════════════════════════════════════════════════

  // Create Pinecone client — authenticates with your API key
  // Takes: { apiKey: string }
  // Returns: Pinecone client object
  const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });

  // Get a reference to your specific index (like a "table" in a database)
  // Takes: index name (string from .env)
  // Returns: Index object you can query/upsert to
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  // ═══════════════════════════════════════════════════════════════════════════
  // NEW: Create PineconeStore instance, then add documents
  //
  // PineconeStore constructor just sets up the connection:
  //   Takes: (embeddings object, { pineconeIndex })
  //   Returns: PineconeStore instance (a vector store)
  //
  // Then addDocuments() does the heavy lifting:
  //   1. Takes each document's pageContent
  //   2. Calls embeddings.embedDocuments() to convert text → vectors
  //   3. Uploads vectors + metadata to Pinecone
  //
  // Takes: Document[]
  // Returns: void (side effect: data is stored in Pinecone)
  // ═══════════════════════════════════════════════════════════════════════════
  
  console.log('🧠 Embedding and storing chunks in Pinecone...');
  
  const vectorStore = new PineconeStore(embeddings, {
    pineconeIndex,       // Which Pinecone index to store in
    maxConcurrency: 5,   // Max 5 parallel uploads (prevents overwhelming the API)
  });

  // addDocuments() = embed all chunks + upload to Pinecone
  // Internally calls: embeddings.embedDocuments(texts) → pineconeIndex.upsert(vectors)
  await vectorStore.addDocuments(cleanDocs);

  // ════════════════════════════════════════════════════════════════════════
  // OLD APPROACH (commented out):
  // ════════════════════════════════════════════════════════════════════════
  // await PineconeStore.fromDocuments(cleanDocs, embeddings, {
  //   pineconeIndex,
  //   maxConcurrency: 5,
  // });
  //
  // fromDocuments() was a static factory method that did everything in one call:
  //   1. Created a new PineconeStore
  //   2. Embedded all documents
  //   3. Uploaded to Pinecone
  //   4. Returned the store instance
  //
  // The new approach (constructor + addDocuments) is more flexible:
  //   - You can add documents in batches
  //   - You can reuse the same store instance for both indexing AND querying
  //   - You can call addDocuments() multiple times for different PDFs
  // ════════════════════════════════════════════════════════════════════════

  console.log('✅ Done! All chunks embedded and stored in Pinecone.');
}

// Run the indexing function
indexing();