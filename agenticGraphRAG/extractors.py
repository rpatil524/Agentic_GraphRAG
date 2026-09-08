"""LLM-assisted extraction of latent actors from SHAB Event text."""

import json
import os

import requests


def batch_real_llm_extractor_openai(batch_items, api_key):
    """
    Extract weak actors from a batch of Event texts using temporary integer IDs.

    Integer IDs reduce prompt size and prevent the model from altering complex
    publication identifiers. The returned keys are mapped back to Event UIDs.
    """
    if not batch_items:
        return {}
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for weak-node extraction.")

    # Map compact prompt IDs back to the original Event UIDs.
    id_map = {str(i): item['id'] for i, item in enumerate(batch_items)}

    combined_text = ""
    for i, item in enumerate(batch_items):
        # Match the 2,000-character extraction window documented in the paper.
        clean_content = str(item['text']).replace('\n', ' ')[:2000]
        combined_text += f"\n--- ITEM {i} ---\n{clean_content}\n"

    system_instruction = """
    You are an expert Data Extraction AI for Swiss Commercial Registry data.
    You will receive multiple texts, numbered 0, 1, 2, etc.
    
    TASK: For EACH text, extract 'Person' (natural persons) and 'Organization' (companies).
    
    OUTPUT FORMAT:
    Return a single JSON OBJECT where keys are the ITEM NUMBERS (Strings) and values are lists of extracted entities.
    Example:
    {
      "0": [{"name": "Hans Meier", "type": "Person", "role": "Director"}],
      "1": []
    }

    RULES:
    1. Valid JSON only.
    2. Split names (e.g., "A, B" -> two objects).
    3. Disambiguate generic authorities (append city if known).
    4. Ignore the publisher itself (SHAB, SOGC).
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = {
        "model": os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini-2024-07-18"),
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Here are the texts to analyze:\n{combined_text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        
        result_json = response.json()

        if 'choices' not in result_json or not result_json['choices']:
            raise ValueError("OpenAI API returned no choices.")

        message = result_json['choices'][0].get('message', {})
        content = message.get('content')

        if not content:
            raise ValueError("OpenAI API returned empty content.")

        try:
            batch_results = json.loads(content)
        except json.JSONDecodeError:
            clean_text = content.replace("```json", "").replace("```", "").strip()
            batch_results = json.loads(clean_text)

        final_results = {}
        for simple_id, entities in batch_results.items():
            original_real_id = id_map.get(str(simple_id))
            if original_real_id:
                if not isinstance(entities, list):
                    final_results[original_real_id] = []
                else:
                    final_results[original_real_id] = _standardize_entities(entities)

        missing_ids = set(id_map.values()) - set(final_results)
        if missing_ids:
            raise ValueError(
                "OpenAI extraction omitted required event IDs: "
                + ", ".join(sorted(str(item) for item in missing_ids))
            )

        return final_results

    except Exception as e:
        print(f"   [!] OpenAI Batch Error: {e}")
        return {}


def _standardize_entities(entities):
    """Normalize LLM fields and filter gazette publisher names."""
    if not entities or not isinstance(entities, (list, dict)):
        return []
    
    standardized = []
    
    if isinstance(entities, dict):
        for key in entities:
            if isinstance(entities[key], list):
                entities = entities[key]
                break

    if not isinstance(entities, list):
        return []

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get('name') or ent.get('entity_name')
        etype = ent.get('type') or ent.get('entity_type')
        role = ent.get('role')
        
        if isinstance(name, list):
            name = " ".join(str(item) for item in name)
        if isinstance(role, list):
            role = ", ".join(str(item) for item in role)
        
        if name and etype:
            name = str(name).strip()
            role = str(role).strip() if role else "Mentioned"
            etype_clean = 'Person' if 'person' in str(etype).lower() else 'Organization'
            
            if name.upper() in {'SHAB', 'SOGC', 'FOSC', 'KAB', 'KANT', 'AMTSBLATT'}:
                continue

            standardized.append({
                'name': name,
                'type': etype_clean, 
                'role': role,
                'confidence': 'High (LLM)'
            })
            
    return standardized
