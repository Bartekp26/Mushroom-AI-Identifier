"""Configuration settings for the Mushroom Identification app."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("API_KEY")

# Model Configuration
CNN_MODEL_PATH = "Model/mushroomCNNclasifier.h5"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# Knowledge Base Configuration
KNOWLEDGE_BASE_FILES = [
    "Knowledge_base/wild_food_uk.json",
    "Knowledge_base/mushroom_world.json",
    "Knowledge_base/wikipedia.json",
    "Knowledge_base/others.json"
]
EMBEDDINGS_PATH = "Knowledge_base/embeddings.npy"

# Image Processing Configuration
IMAGE_SIZE = (224, 224)
TEMP_IMAGE_PATH = "temp/temp_img.jpg"

# RAG Configuration
TOP_K_DOCUMENTS = 3
CONFIDENCE_THRESHOLD = 0.90

# Mushroom Species List
MUSHROOM_SPECIES = [
    'Agaricus augustus', 'Agaricus xanthodermus', 'Amanita amerirubescens', 'Amanita augusta',
    'Amanita brunnescens', 'Amanita calyptroderma', 'Amanita citrina', 'Amanita flavoconia',
    'Amanita muscaria', 'Amanita pantherina', 'Amanita persicina', 'Amanita phalloides',
    'Amanita rubescens', 'Amanita velosa', 'Apioperdon pyriforme', 'Armillaria borealis',
    'Armillaria mellea', 'Armillaria tabescens', 'Artomyces pyxidatus', 'Bjerkandera adusta',
    'Bolbitius titubans', 'Boletus edulis', 'Boletus pallidus', 'Boletus reticulatus',
    'Boletus rex-veris', 'Calocera viscosa', 'Calycina citrina', 'Cantharellus californicus',
    'Cantharellus cibarius', 'Cantharellus cinnabarinus', 'Cerioporus squamosus', 'Cetraria islandica',
    'Chlorociboria aeruginascens', 'Chlorophyllum brunneum', 'Chlorophyllum molybdites',
    'Chondrostereum purpureum', 'Cladonia fimbriata', 'Cladonia rangiferina', 'Cladonia stellaris',
    'Clitocybe nebularis', 'Clitocybe nuda', 'Coltricia perennis', 'Coprinellus disseminatus',
    'Coprinellus micaceus', 'Coprinopsis atramentaria', 'Coprinopsis lagopus', 'Coprinus comatus',
    'Crucibulum laeve', 'Cryptoporus volvatus', 'Daedaleopsis confragosa', 'Daedaleopsis tricolor',
    'Entoloma abortivum', 'Evernia mesomorpha', 'Evernia prunastri', 'Flammulina velutipes',
    'Fomes fomentarius', 'Fomitopsis betulina', 'Fomitopsis mounceae', 'Fomitopsis pinicola',
    'Galerina marginata', 'Ganoderma applanatum', 'Ganoderma curtisii', 'Ganoderma oregonense',
    'Ganoderma tsugae', 'Gliophorus psittacinus', 'Gloeophyllum sepiarium', 'Graphis scripta',
    'Grifola frondosa', 'Gymnopilus luteofolius', 'Gyromitra esculenta', 'Gyromitra gigas',
    'Gyromitra infula', 'Hericium coralloides', 'Hericium erinaceus', 'Hygrophoropsis aurantiaca',
    'Hypholoma fasciculare', 'Hypholoma lateritium', 'Hypogymnia physodes', 'Hypomyces lactifluorum',
    'Imleria badia', 'Inonotus obliquus', 'Ischnoderma resinosum', 'Kuehneromyces mutabilis',
    'Laccaria ochropurpurea', 'Lactarius deliciosus', 'Lactarius torminosus', 'Lactarius turpis',
    'Laetiporus sulphureus', 'Leccinum albostipitatum', 'Leccinum aurantiacum', 'Leccinum scabrum',
    'Leccinum versipelle', 'Lepista nuda', 'Leratiomyces ceres', 'Leucoagaricus americanus',
    'Leucoagaricus leucothites', 'Lobaria pulmonaria', 'Lycogala epidendrum', 'Lycoperdon perlatum',
    'Lycoperdon pyriforme', 'Macrolepiota procera', 'Merulius tremellosus', 'Mutinus ravenelii',
    'Mycena haematopus', 'Mycena leaiana', 'Nectria cinnabarina', 'Omphalotus illudens',
    'Omphalotus olivascens', 'Panaeolus papilionaceus', 'Panellus stipticus', 'Parmelia sulcata',
    'Paxillus involutus', 'Peltigera aphthosa', 'Peltigera praetextata', 'Phaeolus schweinitzii',
    'Phaeophyscia orbicularis', 'Phallus impudicus', 'Phellinus igniarius', 'Phellinus tremulae',
    'Phlebia radiata', 'Phlebia tremellosa', 'Pholiota aurivella', 'Pholiota squarrosa',
    'Phyllotopsis nidulans', 'Physcia adscendens', 'Platismatia glauca', 'Pleurotus ostreatus',
    'Pleurotus pulmonarius', 'Psathyrella candolleana', 'Pseudevernia furfuracea',
    'Pseudohydnum gelatinosum', 'Psilocybe azurescens', 'Psilocybe caerulescens', 'Psilocybe cubensis',
    'Psilocybe cyanescens', 'Psilocybe ovoideocystidiata', 'Psilocybe pelliculosa',
    'Retiboletus ornatipes', 'Rhytisma acerinum', 'Sarcomyxa serotina', 'Sarcoscypha austriaca',
    'Sarcosoma globosum', 'Schizophyllum commune', 'Stereum hirsutum', 'Stereum ostrea',
    'Stropharia aeruginosa', 'Stropharia ambigua', 'Suillus americanus', 'Suillus granulatus',
    'Suillus grevillei', 'Suillus luteus', 'Suillus spraguei', 'Tapinella atrotomentosa',
    'Trametes betulina', 'Trametes gibbosa', 'Trametes hirsuta', 'Trametes ochracea',
    'Trametes versicolor', 'Tremella mesenterica', 'Trichaptum biforme', 'Tricholoma murrillianum',
    'Tricholomopsis rutilans', 'Tylopilus felleus', 'Tylopilus rubrobrunneus', 'Urnula craterium',
    'Verpa bohemica', 'Volvopluteus gloiocephalus', 'Vulpicida pinastri', 'Xanthoria parietina'
]