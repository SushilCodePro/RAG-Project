import readlineSync from 'readline-sync';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'; // ✅ kept — chat still works
import { GoogleGenAI } from '@google/genai';                      // ✅ ADDED: for embeddings only
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';
dotenv.config();
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';

// ─── CHANGED: embeddings — replaced GoogleGenerativeAIEmbeddings (broken) ────
// GoogleGenerativeAIEmbeddings from @langchain/google-genai is broken in JS
// embedQuery was returning wrong dimensions, so we call Google SDK directly
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function embedQuery(text) {
  console.log('🔍 Embedding query...');
  const res = await genAI.models.embedContent({
    model: 'gemini-embedding-001',
    contents: text.trim(),
    config: { taskType: 'RETRIEVAL_QUERY' }, // ✅ QUERY not DOCUMENT for search
  });
  console.log(`✅ Query embedded. Dimensions: ${res.embeddings[0].values.length}`);
  return res.embeddings[0].values;
}

// ─── KEPT: ChatGoogleGenerativeAI works fine for chat ─────────────────────────
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: 'gemini-2.5-flash',  // ✅ changed from 2.5-flash (preview) to stable
  temperature: 0.3,
});

// ─── CHANGED: added apiKey explicitly (Pinecone() without it can fail) ────────
const pineconeIndex = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
}).Index(process.env.PINECONE_INDEX_NAME);

async function chatting(question) {
  try {
    console.log(`\n💬 Question: "${question}"`);

    // Step 1: e mbed the question
    const queryVector = await embedQuery(question);

    // Step 2: search Pinecone for top 10 similar chunks
    console.log('🌲 Searching Pinecone...');
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });
    console.log(`✅ Found ${searchResults.matches.length} matches`);

    // Step 3: build context from search results
    // metadata.text is what we stored during indexing
    const context = searchResults.matches
      .map(match => match.metadata.text)
      .join('\n\n---\n\n');

    if (!context.trim()) {
      console.log('⚠️  No context found in Pinecone. Is your index populated?');
      return;
    }

    // Step 4: build prompt + run LLM chain
    console.log('🤖 Asking Gemini...');
    const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the answer is not in the context, say "I don't have enough information to answer that question."
- Be concise and clear
- Use code examples from the context if relevant

Answer:
    `);

    const chain = RunnableSequence.from([
      promptTemplate,
      model,
      new StringOutputParser(),
    ]);

    const answer = await chain.invoke({ context, question });

    console.log('\n─────────────────────────────');
    console.log(answer);
    console.log('─────────────────────────────\n');

  } catch (err) {
    console.error('❌ Error in chatting():', err.message);
    if (err.status) console.error('HTTP Status:', err.status);
  }
}

async function main() {
  const userQuestion = readlineSync.question('Ask me anything --> ');
  await chatting(userQuestion);
  main(); // loop
}
 
main();