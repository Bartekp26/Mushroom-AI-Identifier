import google.genai as genai
from google.genai.types import GenerateContentConfig
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class MushroomRAGAgent:
    def __init__(self, api_key: str, knowledge_base: List[str],
                 model_name: str, embedding_model: str, embeddings: np.ndarray = None):

        self.system_instructions = """You are an expert mycologist - a mushroom specialist.

            CRITICAL RULES:
            1. Answer ONLY based on the provided MUSHROOM KNOWLEDGE BASE documents
            2. If information is not in the provided documents, clearly state: "This information is not available in my knowledge base"
            3. For POISONOUS/TOXIC mushrooms, emphasize warnings STRONGLY with emojis (🚨⚠️💀)
            4. Never make up information - only use what's in the provided documents
            5. If the species is not specified in question answer about primary species or alternative species

            OUTPUT FORMAT:
            - For the FIRST identification query: Use the structured format below
            - For follow-up questions: Respond conversationally and concisely


            STRUCTURED FORMAT (first query only):
            Name: [Common Name] ([Scientific Name])
            Confidence: [Confidence]

            Safety Status:
            ✅ EDIBLE / ⚠️ POISONOUS / 💀 DEADLY POISONOUS
            [Brief safety note]

            Key Identification Features:
            - Cap: [description]
            - Stem: [description]
            - Gills/Pores: [description]
            - Distinctive traits: [unique features]

            Location, Habitat & Season:
            - Geographic range: [location]
            - Habitat: [where it grows]
            - Season: [when it appears]

            Look-alikes:
            - [Any dangerous similar species]

            Alternative predictions:
            - [All alternative predictions]

            Keep all descriptions very concise - only few words per point.
        """

       
        self.online_mode = False
        self.client = None
        self.chat = None
        self.model_name = model_name

        # Try if api is avaible
        if api_key:
            try:
                # Configure Gemini with new API
                self.client = genai.Client(api_key=api_key)
                # Initialize chat with automatic history management ✅
                self.chat = self.client.chats.create(
                    model=model_name,
                    config=GenerateContentConfig(system_instruction=self.system_instructions)
                )
                self.online_mode = True
                print("Połączono z Gemini API.")
            except Exception as e:
                print(f"Problem z połączeniem lub kluczem API ({e}).")
        else:
            print("Brak klucza API.")

       
        # Knowledge base
        self.knowledge_base = knowledge_base

        # Load semantic embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Create semantic embeddings for documents
        if embeddings is None:
            self.doc_embeddings = self.embedding_model.encode(knowledge_base)
        else:
            self.doc_embeddings = embeddings

        # Track current identification
        self.current_identification = None

        # Track first retrieved documents
        self.first_retrieved_docs = None


    def _retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        # Create embedding for the query
        query_embedding = self.embedding_model.encode(query)

        # Calculate semantic similarity with all documents
        similarities = []
        for doc_embedding in self.doc_embeddings:
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(float(sim))
        similarities = np.array(similarities)

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "document": self.knowledge_base[idx],
                    "similarity": float(similarities[idx]),
                    "index": int(idx)
                })
        return results


    def _build_context(self, relevant_docs: List[Dict], query: str) -> str:
        context_parts = [
            f"{self.system_instructions}\n",
            "\n=== MUSHROOM KNOWLEDGE BASE (Retrieved Documents) ===\n"
        ]

        for i, doc_data in enumerate(relevant_docs, 1):
            doc = doc_data["document"]
            score = doc_data["similarity"]

            context_parts.append(f"\n--- Document {i} (relevance: {score:.3f}) ---")
            context_parts.append(doc)
            context_parts.append("---\n")

        context_parts.append(f"\n=== USER QUESTION ===")
        context_parts.append(f"{query}\n")
        context_parts.append("\nYour response (following all rules):")

        return "\n".join(context_parts)

    def send_message(self, user_message: str, top_k: int = 3, verbose: bool = False) -> str:
        # Block if there is no internet connection
        if not self.online_mode:
            return "No internet connection. Please connect to the internet to chat with the mycologist assistant."

        relevant_docs = self._retrieve_relevant_docs(user_message, top_k=top_k)

        # ADD CONTEXT FROM PREVIOUS QUERY
        current_indices = {doc['index'] for doc in relevant_docs}
        for prev_doc in self.first_retrieved_docs:
            if prev_doc['index'] not in current_indices:
                relevant_docs.append(prev_doc)

        # AUGMENTATION
        context = self._build_context(relevant_docs, user_message)

        if verbose:
            print(f"\n🔍 Found {len(relevant_docs)} relevant documents:")
            for doc_data in relevant_docs:
                doc_preview = doc_data['document'][:80].replace('\n', ' ')
                print(f"  - {doc_preview}... (score: {doc_data['similarity']:.3f})")

            print(f"\n📝 Context length: {len(context)} characters")
            history = self.chat.get_history()
            print(f"💬 Chat history length: {len(history)} messages")

        # GENERATION (chat automatically handles history!) ✅
        try:
            response = self.chat.send_message(message=context)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        if self.online_mode and self.client:
            self.chat = self.client.chats.create(model=self.model_name)
        self.first_retrieved_docs = []
        print("✓ Conversation history cleared")

    def get_history(self) -> List:
        if self.online_mode and self.chat:
            return self.chat.get_history()
        return []


    def initialize_from_predictions(self, predictions: Dict[str, float], verbose: bool = False) -> str:

        # Sort by confidence
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        primary_species, primary_conf = sorted_predictions[0]

        # If there is not internet connection print predictions and warning
        if not self.online_mode:
            alts_list = [f"- {s} (Confidence: {c:.2%})" for s, c in sorted_predictions[1:]]
            alts = "\n".join(alts_list)
            return f"""[OFFLINE MODE - NO INTERNET CONNECTION]

PRIMARY PREDICTION:
Name: {primary_species}
Confidence: {primary_conf:.2%}

ALTERNATIVE PREDICTIONS:
{alts if alts else "None"}

⚠️ NOTE: Information from the Knowledge Base is unavailable.
Please connect to the internet to receive a detailed identification card,
safety warnings, and look-alike analysis."""
        
        # Retrieve documents for predicted species

        relevant_docs = []
        for species, confidence in sorted_predictions:
            relevant_docs.extend(self._retrieve_relevant_docs(species, top_k=2))

        self.first_retrieved_docs = relevant_docs

        if verbose:
            print(f"\n🔍 Retrieved documents for all candidates:")
            for doc_data in relevant_docs[:5]:
                doc_preview = doc_data['document'][:80].replace('\n', ' ')
                print(f"  - {doc_preview}... (score: {doc_data['similarity']:.3f})")

        # Build context with special instructions
        context = self._build_identification_context(relevant_docs, sorted_predictions)

        # Generate response using chat (maintains history automatically)
        try:
            response = self.chat.send_message(message=context)

            #Store current identification
            self.current_identification = {
                'primary': sorted_predictions[0],
                'alternatives': sorted_predictions[1:],
            }

            return response.text
        except Exception as e:
            return f"Error: {str(e)}"


    def _build_identification_context(self, relevant_docs: List[Dict],
                                      predictions: List[Tuple[str, float]]) -> str:

        primary_species, primary_conf = predictions[0]

        context_parts = [
            f"{self.system_instructions}\n",
            "\n=== COMPUTER VISION IDENTIFICATION RESULTS ===\n",
            f"PRIMARY PREDICTION: {primary_species} (Confidence: {primary_conf:.2%})\n",
        ]

        if len(predictions) > 1:
            context_parts.append("ALTERNATIVE PREDICTIONS:")
            for i, (species, conf) in enumerate(predictions[1:], 2):
                context_parts.append(f"{species} (Confidence: {conf:.2%})")

        context_parts.append("\n=== MUSHROOM KNOWLEDGE BASE ===\n")

        for i, doc_data in enumerate(relevant_docs, 1):
            doc = doc_data["document"]
            score = doc_data["similarity"]

            context_parts.append(f"\n--- Document {i} (relevance: {score:.3f}) ---")
            context_parts.append(doc)
            context_parts.append("---\n")

        context_parts.append("\n=== TASK ===")
        context_parts.append(
            f"Provide a detailed identification card for {primary_species} "
            f"(the primary prediction with {primary_conf:.2%} confidence).\n"
        )

        if primary_conf < 0.90:
            context_parts.append(
                f"""Tell the user this warning:
⚠️ Confidence: {primary_conf:.2%} - EXPERT VERIFICATION REQUIRED

🔍 For better identification, please provide additional photos:
- Cap underside (gills/pores)
- Full stem with base
- Growing habitat and surroundings

Clear photos from multiple angles help distinguish similar species.""")

        if len(predictions) > 1:
            context_parts.append(
                "\n⚠️ IMPORTANT: Also check if any alternative predictions "
                f"({', '.join([s for s, _ in predictions[1:]])}) are dangerous species. "
                "If so, add a warning section at the very beginning before identification card and inform that this alternative species is in predictions.\n"
            )

        return "\n".join(context_parts)