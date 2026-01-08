from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import google.genai as genai
from google.genai.types import GenerateContentConfig
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def get_api_key():
    with open('keys.env', 'r', encoding='utf-8') as f:
        for line in f:
            if 'API_KEY=' in line:
                return line.strip().split('=', 1)[1]
    return None

def prepare_knowledge_base(file_paths):
    knowledge_base = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = []

        for mushroom_name, mushroom_info in data.items():
            doc_parts = [mushroom_name]
            for label, value in mushroom_info.items():
                doc_parts.append(f"{label}: {value}")

            full_document = "\n".join(doc_parts)
            documents.append(full_document)

        knowledge_base.extend(documents)
    return knowledge_base



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

            Keep all descriptions very concise - only few words per point.
        """

        # Configure Gemini with new API
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Initialize chat with automatic history management ✅
        self.chat = self.client.chats.create(model=model_name,
                                             config=GenerateContentConfig(system_instruction=self.system_instructions))

        # Knowledge base
        self.knowledge_base = knowledge_base

        # Load semantic embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Create semantic embeddings for documents
        if embeddings is None:
            self.doc_embeddings = self.embedding_model.encode(knowledge_base)
        else:
            self.doc_embeddings = embeddings

        # Track turns
        self.turn = 0

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
            self.turn += 1
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        self.chat = self.client.chats.create(model=self.model_name)
        self.first_retrieved_docs = []
        self.turn = 0
        print("✓ Conversation history cleared")

    def get_history(self) -> List:
        return self.chat.get_history()


    def initialize_from_predictions(self, predictions: Dict[str, float], verbose: bool = False) -> str:

        # Sort by confidence
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Retrieve documents for predicted species
        relevant_docs = []
        for species, confidence in sorted_predictions:
            relevant_docs.extend(self._retrieve_relevant_docs(species, top_k=2))

        if self.turn == 0:
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
            self.turn += 1

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

                Clear photos from multiple angles help distinguish similar species."""
            )

        if len(predictions) > 1:
            context_parts.append(
                "\n⚠️ IMPORTANT: Also check if any alternative predictions "
                f"({', '.join([s for s, _ in predictions[1:]])}) are dangerous species. "
                "If so, add a warning section at the very beginning before identification card and inform that this species is in predictions.\n"
            )

        return "\n".join(context_parts)
    
mushroom_species = ['Agaricus augustus', 'Agaricus xanthodermus', 'Amanita amerirubescens', 'Amanita augusta', 'Amanita brunnescens', 'Amanita calyptroderma', 'Amanita citrina', 'Amanita flavoconia', 'Amanita muscaria', 'Amanita pantherina', 'Amanita persicina', 'Amanita phalloides', 'Amanita rubescens', 'Amanita velosa', 'Apioperdon pyriforme', 'Armillaria borealis', 'Armillaria mellea', 'Armillaria tabescens', 'Artomyces pyxidatus', 'Bjerkandera adusta', 'Bolbitius titubans', 'Boletus edulis', 'Boletus pallidus', 'Boletus reticulatus', 'Boletus rex-veris', 'Calocera viscosa', 'Calycina citrina', 'Cantharellus californicus', 'Cantharellus cibarius', 'Cantharellus cinnabarinus', 'Cerioporus squamosus', 'Cetraria islandica', 'Chlorociboria aeruginascens', 'Chlorophyllum brunneum', 'Chlorophyllum molybdites', 'Chondrostereum purpureum', 'Cladonia fimbriata', 'Cladonia rangiferina', 'Cladonia stellaris', 'Clitocybe nebularis', 'Clitocybe nuda', 'Coltricia perennis', 'Coprinellus disseminatus', 'Coprinellus micaceus', 'Coprinopsis atramentaria', 'Coprinopsis lagopus', 'Coprinus comatus', 'Crucibulum laeve', 'Cryptoporus volvatus', 'Daedaleopsis confragosa', 'Daedaleopsis tricolor', 'Entoloma abortivum', 'Evernia mesomorpha', 'Evernia prunastri', 'Flammulina velutipes', 'Fomes fomentarius', 'Fomitopsis betulina', 'Fomitopsis mounceae', 'Fomitopsis pinicola', 'Galerina marginata', 'Ganoderma applanatum', 'Ganoderma curtisii', 'Ganoderma oregonense', 'Ganoderma tsugae', 'Gliophorus psittacinus', 'Gloeophyllum sepiarium', 'Graphis scripta', 'Grifola frondosa', 'Gymnopilus luteofolius', 'Gyromitra esculenta', 'Gyromitra gigas', 'Gyromitra infula', 'Hericium coralloides', 'Hericium erinaceus', 'Hygrophoropsis aurantiaca', 'Hypholoma fasciculare', 'Hypholoma lateritium', 'Hypogymnia physodes', 'Hypomyces lactifluorum', 'Imleria badia', 'Inonotus obliquus', 'Ischnoderma resinosum', 'Kuehneromyces mutabilis', 'Laccaria ochropurpurea', 'Lactarius deliciosus', 'Lactarius torminosus', 'Lactarius turpis', 'Laetiporus sulphureus', 'Leccinum albostipitatum', 'Leccinum aurantiacum', 'Leccinum scabrum', 'Leccinum versipelle', 'Lepista nuda', 'Leratiomyces ceres', 'Leucoagaricus americanus', 'Leucoagaricus leucothites', 'Lobaria pulmonaria', 'Lycogala epidendrum', 'Lycoperdon perlatum', 'Lycoperdon pyriforme', 'Macrolepiota procera', 'Merulius tremellosus', 'Mutinus ravenelii', 'Mycena haematopus', 'Mycena leaiana', 'Nectria cinnabarina', 'Omphalotus illudens', 'Omphalotus olivascens', 'Panaeolus papilionaceus', 'Panellus stipticus', 'Parmelia sulcata', 'Paxillus involutus', 'Peltigera aphthosa', 'Peltigera praetextata', 'Phaeolus schweinitzii', 'Phaeophyscia orbicularis', 'Phallus impudicus', 'Phellinus igniarius', 'Phellinus tremulae', 'Phlebia radiata', 'Phlebia tremellosa', 'Pholiota aurivella', 'Pholiota squarrosa', 'Phyllotopsis nidulans', 'Physcia adscendens', 'Platismatia glauca', 'Pleurotus ostreatus', 'Pleurotus pulmonarius', 'Psathyrella candolleana', 'Pseudevernia furfuracea', 'Pseudohydnum gelatinosum', 'Psilocybe azurescens', 'Psilocybe caerulescens', 'Psilocybe cubensis', 'Psilocybe cyanescens', 'Psilocybe ovoideocystidiata', 'Psilocybe pelliculosa', 'Retiboletus ornatipes', 'Rhytisma acerinum', 'Sarcomyxa serotina', 'Sarcoscypha austriaca', 'Sarcosoma globosum', 'Schizophyllum commune', 'Stereum hirsutum', 'Stereum ostrea', 'Stropharia aeruginosa', 'Stropharia ambigua', 'Suillus americanus', 'Suillus granulatus', 'Suillus grevillei', 'Suillus luteus', 'Suillus spraguei', 'Tapinella atrotomentosa', 'Trametes betulina', 'Trametes gibbosa', 'Trametes hirsuta', 'Trametes ochracea', 'Trametes versicolor', 'Tremella mesenterica', 'Trichaptum biforme', 'Tricholoma murrillianum', 'Tricholomopsis rutilans', 'Tylopilus felleus', 'Tylopilus rubrobrunneus', 'Urnula craterium', 'Verpa bohemica', 'Volvopluteus gloiocephalus', 'Vulpicida pinastri', 'Xanthoria parietina']
model_loaded = load_model(".\mushroomCNNclasifier.h5")

def get_predictions(img_path: str, top_k: int = 3):
    # Prepare image to prediction
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model_loaded.predict(img_array)
    predictions_flat = predictions.ravel()

    # Top k predictions
    indexes = np.argpartition(predictions_flat, -top_k)[-top_k:]
    values = predictions_flat[indexes]

    # Sort descending
    sorted = np.argsort(values)[::-1]
    indexes, values = indexes[sorted], values[sorted]
    species = [mushroom_species[i] for i in indexes]

    predictions_dict = dict(zip(species, values))

    return predictions_dict


