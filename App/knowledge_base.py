import json

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