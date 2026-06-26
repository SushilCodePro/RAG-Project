import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenAI } from '@google/genai';  // ✅ CHANGED: new SDK

// ✅ ADDED: wrapper so PineconeStore can still use embeddings normally
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const embeddings = {
  embedDocuments: async (texts) => {
    const vectors = [];
    for (const text of texts) {
      const res = await genAI.models.embedContent({
        model: 'gemini-embedding-001',  // ✅ CHANGED: new model
        contents: text.trim(),
        config: { taskType: 'RETRIEVAL_DOCUMENT' },
      });
      vectors.push(res.embeddings[0].values);
    }
    return vectors;
  },
  embedQuery: async (text) => {
    const res = await genAI.models.embedContent({
      model: 'gemini-embedding-001',
      contents: text.trim(),
      config: { taskType: 'RETRIEVAL_QUERY' },
    });
    return res.embeddings[0].values;
  },
};

async function indexing() {
  const PDF_PATH = 'data/Node.pdf';
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs = await pdfLoader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

  // ✅ ADDED: filter empty chunks (fixes "dimension 0" error)
  const cleanDocs = chunkedDocs.filter(doc => doc.pageContent.trim().length > 50);
  console.log(`chunks: ${chunkedDocs.length} → after filter: ${cleanDocs.length}`);

  const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY }); // ✅ CHANGED: added apiKey
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  await PineconeStore.fromDocuments(cleanDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });

  console.log('✅ Done!');
}

indexing();