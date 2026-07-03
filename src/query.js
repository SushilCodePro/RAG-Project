// ============================================================================
// query.js — The QUERYING pipeline for RAG
//
// PURPOSE: Takes a user's question → searches Pinecone for relevant chunks →
//          sends question + context to Gemini LLM → returns an answer
//
// FLOW DIAGRAM:
//   User types a question
//     → [PineconeStore.similaritySearch()] Embeds the question & searches Pinecone
//     → Gets back the most relevant document chunks
//     → [PromptTemplate] Combines question + chunks into a prompt
//     → [ChatGoogleGenerativeAI] Sends prompt to Gemini LLM
//     → [StringOutputParser] Extracts the text answer
//     → Prints the answer
//
// PREREQUISITE: Run indexing.js first to populate the Pinecone index!
// ============================================================================

// ─── Step 0: Load environment variables and imports ─────────────────────────

import * as dotenv from 'dotenv';
dotenv.config(); // Loads .env variables (GEMINI_API_KEY, PINECONE_API_KEY, etc.)

// readline-sync: Lets us read user input from the terminal (synchronously)
// Takes: prompt string  →  Returns: whatever the user types
import readlineSync from 'readline-sync';

// ChatGoogleGenerativeAI: LangChain's wrapper for Gemini chat models
// Takes: { apiKey, model, temperature }  →  Returns: a chat model you can .invoke()
// This is for GENERATING ANSWERS (not embeddings)
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

// GoogleGenAI: Google's direct SDK for embeddings
// We still need this for the custom embeddings wrapper
import { GoogleGenAI } from '@google/genai';

// Pinecone: Client SDK to connect to Pinecone vector database
import { Pinecone } from '@pinecone-database/pinecone';

// PineconeStore: LangChain's vector store wrapper for Pinecone
// NEW: We now use this for QUERYING too (not just indexing)
// It has .similaritySearch() which handles embedding + searching automatically
import { PineconeStore } from '@langchain/pinecone';

// PromptTemplate: Creates a reusable template with {variables}
// Takes: template string with {placeholders}  →  Returns: a Runnable
// When invoked, replaces {context} and {question} with actual values
import { PromptTemplate } from '@langchain/core/prompts';

// StringOutputParser: Extracts plain text from LLM response
// LLM returns a complex message object → this parser pulls out just the text string
// Takes: AIMessage  →  Returns: string
import { StringOutputParser } from '@langchain/core/output_parsers';

// RunnableSequence: Chains multiple steps together (like a pipeline)
// Step 1 output → becomes Step 2 input → becomes Step 3 input → ...
// Takes: array of Runnables  →  Returns: one combined Runnable
import { RunnableSequence } from '@langchain/core/runnables';


// ─── Step 1: Set up Embeddings (same wrapper as indexing.js) ────────────────
//
// We need embeddings for QUERYING too because:
//   1. User asks: "What is Node.js?"
//   2. We convert that question → vector (numbers)
//   3. We search Pinecone for document vectors closest to that question vector
//   4. "Closest" = most semantically similar content
//
// PineconeStore.similaritySearch() calls embeddings.embedQuery() internally,
// so we don't have to manually embed the question anymore!
// ────────────────────────────────────────────────────────────────────────────

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const embeddings = {
  // embedDocuments: Not used during querying, but PineconeStore expects it
  embedDocuments: async (texts) => {
    const vectors = [];
    for (const text of texts) {
      const res = await genAI.models.embedContent({
        model: 'gemini-embedding-001',
        contents: text.trim(),
        config: { taskType: 'RETRIEVAL_DOCUMENT' },
      });
      vectors.push(res.embeddings[0].values);
    }
    return vectors;
  },

  // embedQuery: Called by similaritySearch() to convert your question → vector
  // Takes: text (string) — the user's question
  // Returns: number[] — a 3072-dimension vector
  embedQuery: async (text) => {
    const res = await genAI.models.embedContent({
      model: 'gemini-embedding-001',
      contents: text.trim(),
      config: { taskType: 'RETRIEVAL_QUERY' },
    });
    return res.embeddings[0].values;
  },
};


// ─── Step 2: Set up the Chat Model (Gemini) ─────────────────────────────────
//
// This is the LLM that GENERATES the final answer.
// It receives: the user's question + relevant context from Pinecone
// It returns: a human-readable answer
//
// ChatGoogleGenerativeAI wraps Google's Gemini API for use in LangChain chains.
// ────────────────────────────────────────────────────────────────────────────

const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-2.5-flash',  // Which Gemini model to use for chat
  temperature: 0.3,             // 0 = deterministic, 1 = creative. 0.3 = mostly factual
});


// ─── Step 3: Connect to Pinecone Vector Store (NEW approach) ────────────────
//
// OLD APPROACH (commented out below): We manually called pineconeIndex.query()
//   with raw vectors. This required us to:
//   1. Manually embed the question
//   2. Manually call pinecone SDK
//   3. Manually extract metadata.text from results
//
// NEW APPROACH: Use PineconeStore which does all of that automatically!
//   vectorStore.similaritySearch("question") handles:
//   1. Embedding the question (calls embeddings.embedQuery())
//   2. Searching Pinecone for nearest vectors
//   3. Returning LangChain Document[] objects with pageContent
// ────────────────────────────────────────────────────────────────────────────

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// Create PineconeStore connected to your existing index
// This is the SAME index that indexing.js populated with document chunks
// Takes: (embeddings, { pineconeIndex })
// Returns: PineconeStore instance with .similaritySearch(), .asRetriever(), etc.
const vectorStore = new PineconeStore(embeddings, { pineconeIndex });


// ════════════════════════════════════════════════════════════════════════════
// OLD QUERY APPROACH (commented out):
// ════════════════════════════════════════════════════════════════════════════
//
// // We had a manual embedQuery function:
// // async function embedQuery(text) {
// //   const res = await genAI.models.embedContent({
// //     model: 'gemini-embedding-001',
// //     contents: text.trim(),
// //     config: { taskType: 'RETRIEVAL_QUERY' },
// //   });
// //   return res.embeddings[0].values;
// // }
//
// // And manually queried Pinecone:
// // const queryVector = await embedQuery(question);
// // const searchResults = await pineconeIndex.query({
// //   topK: 10,
// //   vector: queryVector,
// //   includeMetadata: true,
// // });
// // const context = searchResults.matches
// //   .map(match => match.metadata.text)
// //   .join('\n\n---\n\n');
//
// PROBLEMS WITH OLD APPROACH:
//   1. Had to manually embed the query (extra code)
//   2. Called Pinecone SDK directly (bypassed LangChain's abstraction)
//   3. Had to manually extract .metadata.text from results
//   4. Couldn't use advanced features like MMR (Maximal Marginal Relevance)
//   5. Tightly coupled to Pinecone — switching vector stores meant rewriting
//
// NEW APPROACH: One line does it all:
//   const results = await vectorStore.similaritySearch(question, 10);
// ════════════════════════════════════════════════════════════════════════════


// ─── Step 4: The main chat function ─────────────────────────────────────────

async function chatting(question) {
  try {
    console.log(`\n💬 Question: "${question}"`);

    // ═════════════════════════════════════════════════════════════════════
    // STEP 4a: Search Pinecone for relevant document chunks (NEW approach)
    // ═════════════════════════════════════════════════════════════════════

    // similaritySearch() does 3 things automatically:
    //   1. Calls embeddings.embedQuery(question) → converts question to vector
    //   2. Sends that vector to Pinecone to find the 10 closest document vectors
    //   3. Returns LangChain Document[] objects (not raw Pinecone matches)
    //
    // Takes: (query: string, k: number)
    //   - query: the user's question (plain text, NOT a vector)
    //   - k: how many results to return (top 10 most similar)
    // Returns: Document[] — each has .pageContent (the chunk text) and .metadata

    console.log('🔍 Searching Pinecone for relevant chunks...');
    const results = await vectorStore.similaritySearch(question, 10);
    console.log(`✅ Found ${results.length} relevant chunks`);

    // ═════════════════════════════════════════════════════════════════════
    // ALTERNATIVE: Use similaritySearchWithScore() to also get relevance scores
    // ═════════════════════════════════════════════════════════════════════
    // const resultsWithScores = await vectorStore.similaritySearchWithScore(question, 10);
    // resultsWithScores = [[Document, score], [Document, score], ...]
    // score: 0-1 where 1 = perfect match, 0 = no similarity

    // ═════════════════════════════════════════════════════════════════════
    // ALTERNATIVE: Use asRetriever() for a Runnable (composable in chains)
    // ═════════════════════════════════════════════════════════════════════
    // const retriever = vectorStore.asRetriever({
    //   k: 10,                          // How many results
    //   searchType: "mmr",              // MMR = diverse results (avoids duplicates)
    //   searchKwargs: { fetchK: 20 },   // Fetch 20, then pick 10 most diverse
    // });
    // const results = await retriever.invoke(question);

    // ═════════════════════════════════════════════════════════════════════
    // STEP 4b: Build context string from search results
    // ═════════════════════════════════════════════════════════════════════

    // Combine all chunk texts into one big "context" string
    // Each chunk is separated by --- for readability
    // results[i].pageContent = the actual text of that chunk
    const context = results
      .map(doc => doc.pageContent)    // Extract just the text from each Document
      .join('\n\n---\n\n');            // Join them with separators

    if (!context.trim()) {
      console.log('⚠️  No relevant content found. Is your Pinecone index populated?');
      console.log('   Run: node src/indexing.js first');
      return;
    }

    // ═════════════════════════════════════════════════════════════════════
    // STEP 4c: Create the prompt template
    // ═════════════════════════════════════════════════════════════════════

    // PromptTemplate.fromTemplate() creates a reusable template
    // {context} and {question} are placeholders that get filled when invoked
    //
    // Takes: template string with {variables}
    // Returns: a Runnable — when invoked with { context, question }, 
    //          it returns the filled-in prompt string
    console.log('🤖 Asking Gemini...');
    const promptTemplate = PromptTemplate.fromTemplate(`
          You are a helpful assistant answering questions based on the provided documentation.
          Context from the documentation: {context}
          Question: {question}
          Instructions:
            - Answer the question using ONLY the information from the context above
            - If the answer is not in the context, say "I don't have enough information to answer that question."
            - Be concise and clear
            - Use code examples from the context if relevant
          Answer:
    `);

    // ═════════════════════════════════════════════════════════════════════
    // STEP 4d: Build and run the chain
    // ═════════════════════════════════════════════════════════════════════

    // RunnableSequence.from() creates a pipeline:
    //   Step 1: promptTemplate — fills in {context} and {question}
    //           Input: { context, question }  →  Output: formatted prompt string
    //
    //   Step 2: model — sends the prompt to Gemini LLM
    //           Input: prompt string  →  Output: AIMessage object
    //
    //   Step 3: StringOutputParser — extracts plain text from AIMessage
    //           Input: AIMessage  →  Output: string (the final answer)
    const chain = RunnableSequence.from([
      promptTemplate,          // { context, question } → prompt string
      model,                   // prompt string → AIMessage
      new StringOutputParser(), // AIMessage → plain text string
    ]);

    // chain.invoke() runs all 3 steps in sequence
    // Takes: { context: string, question: string }
    // Returns: string — the LLM's answer
    const answer = await chain.invoke({ context, question });

    console.log('\n─────────────────────────────');
    console.log(answer);
    console.log('─────────────────────────────\n');

  } catch (err) {
    console.error('❌ Error in chatting():', err.message);
    if (err.status) console.error('HTTP Status:', err.status);
  }
}


// ─── Step 5: Interactive loop ───────────────────────────────────────────────
// Keeps asking the user for questions until they close the terminal

async function main() {
  const userQuestion = readlineSync.question('Ask me anything --> ');
  await chatting(userQuestion);
  main(); // Call itself again to keep the loop going
}

main();


// ─────────────────────────────
// I don't have enough information to answer that question. The documentation states that Lesson 3: What is Node.js? includes "a brief tour of the V8 JavaScript engine" 
// and that Lesson 3: Call Stack, Callback Queue, and Event Loop will visualize "how Node.js and V8 manage your asynchronous code," but it does not define what the V8 engine is.
// ─────────────────────────────
