import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from '@google/genai';

// ─── Sleep helper ─────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ─── Custom Embedder with retry + backoff ────────────────────────────────────
class GeminiEmbedder {
  constructor(apiKey) {
    this.client = new GoogleGenAI({ apiKey });
    this.model = 'gemini-embedding-001';
  }

  // Embed single text — retries on 429 automatically
  async embedText(text, retries = 5) {
    const trimmed = text.trim();
    if (!trimmed) return null;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await this.client.models.embedContent({
          model: this.model,
          contents: trimmed,
          config: { taskType: 'RETRIEVAL_DOCUMENT' },
        });
        return response.embeddings[0].values;

      } catch (err) {
        const is429 = err.message?.includes('429') || err.status === 429;

        if (is429 && attempt < retries) {
          // Parse retry delay from error message if available (e.g. "retry in 8.4s")
          const match = err.message?.match(/retry in (\d+(\.\d+)?)s/i);
          const waitSec = match ? Math.ceil(parseFloat(match[1])) + 2 : attempt * 15;
          console.log(`   ⏳ Rate limit hit. Waiting ${waitSec}s before retry (attempt ${attempt}/${retries})...`);
          await sleep(waitSec * 1000);
        } else {
          throw err; // not a 429 or out of retries
        }
      }
    }
  }

  // Embed all docs one by one with delay between each call
  async embedAll(docs, delayMs = 2000) {
    const vectors = [];
    for (let i = 0; i < docs.length; i++) {
      const vec = await this.embedText(docs[i].pageContent);
      vectors.push(vec);

      // Progress log every 10 chunks
      if ((i + 1) % 10 === 0 || i === docs.length - 1) {
        console.log(`   ✅ Embedded ${i + 1}/${docs.length} chunks...`);
      }

      // Delay between every call to stay under rate limit
      if (i < docs.length - 1) {
        await sleep(delayMs);
      }
    }
    return vectors;
  }
}

async function indexing() {
  try {
    // ─── STEP 1: Validate env vars ──────────────────────────────────
    console.log('🔑 Step 1: Checking environment variables...');
    const GEMINI_KEY    = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    const PINECONE_KEY  = process.env.PINECONE_API_KEY;
    const PINECONE_INDEX = process.env.PINECONE_INDEX_NAME;

    if (!GEMINI_KEY)     throw new Error('❌ Missing GEMINI_API_KEY in .env');
    if (!PINECONE_KEY)   throw new Error('❌ Missing PINECONE_API_KEY in .env');
    if (!PINECONE_INDEX) throw new Error('❌ Missing PINECONE_INDEX_NAME in .env');
    console.log('✅ Env vars OK. Index:', PINECONE_INDEX);

    // ─── STEP 2: Load PDF ────────────────────────────────────────────
    const PDF_PATH = './Node.pdf';
    console.log(`\n📄 Step 2: Loading PDF from "${PDF_PATH}"...`);
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log(`✅ PDF loaded. Total pages: ${rawDocs.length}`);

    // ─── STEP 3: Chunk ───────────────────────────────────────────────
    console.log('\n✂️  Step 3: Splitting into chunks...');
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 100,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`✅ Chunking done. Total chunks: ${chunkedDocs.length}`);

    // ─── STEP 3.5: Filter empty chunks ──────────────────────────────
    const cleanDocs = chunkedDocs.filter(doc => {
      const text = doc.pageContent.trim();
      return text.length > 50 && /[a-zA-Z0-9]/.test(text);
    });
    console.log(`🧹 Filtered: ${chunkedDocs.length - cleanDocs.length} empty chunks removed.`);
    console.log(`📦 Clean chunks: ${cleanDocs.length}`);

    // ─── STEP 4: Embed ───────────────────────────────────────────────
    console.log('\n🧠 Step 4: Embedding chunks with Gemini...');
    console.log(`⚠️  Free tier = 1000 req/day. Using 2s delay between calls.`);
    console.log(`⏱️  Estimated time: ~${Math.ceil(cleanDocs.length * 2 / 60)} minutes`);

    const embedder = new GeminiEmbedder(GEMINI_KEY);

    // Test 1 chunk first
    console.log('🧪 Testing 1 chunk...');
    const testVec = await embedder.embedText(cleanDocs[0].pageContent);
    if (!testVec || testVec.length === 0) throw new Error('❌ Test embedding returned empty!');
    console.log(`✅ Test OK! Dimensions: ${testVec.length}`);

    await sleep(2000); // pause after test before bulk

    // Embed all — slow and steady
    const vectors = await embedder.embedAll(cleanDocs, 2000); // 2s between each call

    // Sanity check
    const emptyCount = vectors.filter(v => !v || v.length === 0).length;
    if (emptyCount > 0) throw new Error(`❌ ${emptyCount} empty vectors found!`);
    console.log(`✅ All ${vectors.length} vectors OK! Dimensions: ${vectors[0].length}`);

    // ─── STEP 5: Connect to Pinecone ─────────────────────────────────
    console.log('\n🌲 Step 5: Connecting to Pinecone...');
    const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
    const pineconeIndex = pinecone.Index(PINECONE_INDEX);
    console.log(`✅ Connected to Pinecone index: "${PINECONE_INDEX}"`);

    // ─── STEP 6: Upload to Pinecone ──────────────────────────────────
    console.log(`\n⬆️  Step 6: Uploading ${cleanDocs.length} vectors to Pinecone...`);
    const UPSERT_BATCH = 100;
    for (let i = 0; i < cleanDocs.length; i += UPSERT_BATCH) {
      const batch = cleanDocs.slice(i, i + UPSERT_BATCH).map((doc, j) => ({
        id: `chunk-${i + j}`,
        values: vectors[i + j],
        metadata: {
          text: doc.pageContent,
          source: doc.metadata?.source || PDF_PATH,
          page: doc.metadata?.loc?.pageNumber || doc.metadata?.page || 0,
        },
      }));
      await pineconeIndex.upsert(batch);
      console.log(`   ✅ Upserted ${Math.min(i + UPSERT_BATCH, cleanDocs.length)}/${cleanDocs.length}`);
    }

    console.log('\n🎉 Done! All chunks stored in Pinecone successfully.');

  } catch (err) {
    console.error('\n❌ Error during indexing:');
    console.error('Message :', err.message);
    if (err.status) console.error('HTTP Status:', err.status);
    if (err.cause)  console.error('Cause     :', err.cause);
    process.exit(1);
  }
}

indexing();