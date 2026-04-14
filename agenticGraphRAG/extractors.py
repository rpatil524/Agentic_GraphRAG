import requests
import json
import re

def mock_llm_extractor(event_id, full_text):
    """
    Simulates an AI reading the text and extracting entities.
    Returns a list of 'Weak Nodes' found in the text.
    """
    extracted_entities = []

    # 1. SIMULATION: Detect the "Authority" we found earlier
    # (The AI would naturally pull this out as an Organization)
    if "AUTHORITY_OFFICE:" in full_text:
        match = re.search(r"AUTHORITY_OFFICE: (.*?)(?:\n|$)", full_text)
        if match:
            name = match.group(1).split(',')[0].strip()
            extracted_entities.append({
                'name': name,
                'type': 'Organization',
                'role': 'Authority/Liquidator',
                'confidence': 'High (Structured)'
            })

    # 2. SIMULATION: Detect a specific Person (Testing the Name Hub)
    # Let's pretend we found "Hans Müller" in the text of a specific event
    if "Hans Müller" in full_text: 
        extracted_entities.append({
            'name': 'Hans Müller',
            'type': 'Person',
            'role': 'Mentioned in Text',
            'confidence': 'Medium'
        })

    return extracted_entities

def real_llm_extractor(event_id, full_text):
    """
    The Definitive Extractor.
    Combines strict business rules with robust Python error handling.
    Sends text to local Ollama (Qwen) to extract entities.
    Returns a list of dicts: [{'name': '...', 'type': '...', 'role': '...'}]
    """
    # Wrapper for single item to use the core logic if needed, 
    # but strictly speaking we kept the original implementation below for backward compatibility
    # or simple usage.
    
    # --- 1. THE DETAILED SYSTEM PROMPT ---
    system_instruction = """
    You are an expert Data Extraction AI for Swiss Legal Documents.
    
    TASK: Extract all 'Person' and 'Organization' entities from the text.
    
    CRITICAL RULES:
    1. OUTPUT: Your final answer must be a valid JSON list of objects.
    2. SPLIT NAMES: If multiple people are listed (e.g., "Directors: A, B, and C"), extract them as SEPARATE objects. NEVER combine them into one name.
    3. DISAMBIGUATE AUTHORITIES: If you find a generic authority name like "Notariat" or "Konkursamt", you MUST append the city found in the text (e.g., "Notariat Zürich (Riesbach)").
    4. IGNORE PUBLISHER: Do not extract "SHAB", "SOGC", "FOSC", or "KAB" as entities. These are just the gazette names.
    5. ROLE: Infer the role (e.g., 'Debtor', 'Creditor', 'Liquidator'). Return the role as a single string.
    """
    
    prompt = f"{system_instruction}\n\nTEXT TO ANALYZE:\n{full_text}"

    payload = {
        "model": "qwen3:14b",  
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 2048
        }
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        
        result_text = response.json().get('response', '')
        return _parse_llm_json(result_text, event_id)

    except Exception as e:
        print(f"   [!] Critical Error on Event {event_id}: {e}")
        return []

def batch_real_llm_extractor_ollama(batch_items):
    """
    Processes a batch of texts in a single LLM call using INTEGER MAPPING.
    Args:
        batch_items: List of tuples/dicts [{'id': '...', 'text': '...'}]
    Returns:
        Dict: {original_event_id: [extracted_entities]}
    """
    
    # --- 1. ID MAPPING (The Fix) ---
    # We map simple integers to the real complex IDs to prevent LLM hallucinations.
    # Map: "0" -> "SHAB-2020-..."
    id_map = {str(i): item['id'] for i, item in enumerate(batch_items)}
    
    # --- 2. Construct Batch Prompt with Integers ---
    combined_text = ""
    for i, item in enumerate(batch_items):
        # Truncate slightly to ensure we don't blow the context window if one text is huge
        clean_content = str(item['text']).replace('\n', ' ')[:4000] 
        combined_text += f"\n--- ITEM {i} ---\n{clean_content}\n"
        
    system_instruction = """
    You are an expert Data Extraction AI. You will receive multiple texts, numbered 0, 1, 2, etc.
    
    TASK: For EACH text, extract 'Person' and 'Organization' entities.
    
    OUTPUT FORMAT:
    Return a single JSON OBJECT where keys are the ITEM NUMBERS (Strings) and values are lists of extracted entities.
    Example:
    {
      "0": [{"name": "Hans", "type": "Person", "role": "Director"}],
      "1": [{"name": "Corp AG", "type": "Organization", "role": "Creditor"}]
    }

    RULES:
    1. Valid JSON only.
    2. Split names (A, B -> two objects).
    3. Disambiguate generic authorities (append city).
    4. Ignore publishers (SHAB, SOGC).
    """
    
    prompt = f"{system_instruction}\n\nTEXTS TO ANALYZE:\n{combined_text}"
    
    payload = {
        "model": "qwen3:14b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192  # [RECOMMENDED] Increase this if possible (default is usually 4096)
        }
    }
    
    try:
        # Increase timeout because batches take longer (e.g., 90 seconds)
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90) 
        response.raise_for_status()
        result_text = response.json().get('response', '')
        
        # --- 3. Robust JSON Parsing ---
        clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()
        
        try:
            batch_results = json.loads(clean_text)
        except json.JSONDecodeError:
            # Fallback: Sometimes LLMs add text before/after JSON. Try finding the first '{' and last '}'
            try:
                start = clean_text.find('{')
                end = clean_text.rfind('}') + 1
                if start != -1 and end != -1:
                    batch_results = json.loads(clean_text[start:end])
                else:
                    raise ValueError("No JSON found")
            except Exception:
                print(f"   [!] Batch JSON Decode Error. Raw: {clean_text[:50]}...")
                return {}
            
        # --- 4. Remap & Standardize ---
        final_results = {}
        for simple_id, entities in batch_results.items():
            # Retrieve the ORIGINAL ID using the map
            original_real_id = id_map.get(str(simple_id))
            
            if original_real_id:
                final_results[original_real_id] = _standardize_entities(entities)
            
        return final_results

    except Exception as e:
        print(f"   [!] Batch Error: {e}")
        return {}

def batch_real_llm_extractor_openai(batch_items, api_key):
    """
    Processes a batch of texts using Ollama (Qwen) with INTEGER MAPPING.
    Args:
        batch_items: List of tuples/dicts [{'id': '...', 'text': '...'}]
    Returns:
        Dict: {original_event_id: [extracted_entities]}
    """
    if not batch_items:
        return {}

    # --- 1. ID MAPPING ---
    # Map simple integers "0", "1" to complex IDs to save tokens and reduce hallucination
    id_map = {str(i): item['id'] for i, item in enumerate(batch_items)}
    
    # --- 2. Construct User Content ---
    combined_text = ""
    for i, item in enumerate(batch_items):
        # Truncate to ~2000 chars to save money/tokens (usually enough for header/footer analysis)
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
    
    # --- 3. Prepare OpenAI Payload ---
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",  # Highly recommended for this: fast & cheap
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Here are the texts to analyze:\n{combined_text}"}
        ],
        "response_format": {"type": "json_object"}, # Forces valid JSON output
        "temperature": 0.0,
        "max_tokens": 4096 
    }
    
    try:
        # Timeout set to 60s (OpenAI is usually much faster than local LLMs)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=60
        )
        response.raise_for_status()
        
        # --- 4. Parse OpenAI API Response Structure ---
        result_json = response.json()
        
        # Defensive check against unexpected OpenAI response formats
        if 'choices' not in result_json or not result_json['choices']:
            raise ValueError(f"OpenAI API returned unexpected structure: {result_json}")
            
        message = result_json['choices'][0].get('message', {})
        content = message.get('content')
        
        if not content:
             raise ValueError(f"OpenAI API returned empty content: {result_json}")
        
        # --- 4. Robust Parsing ---
        # OpenAI usually returns clean JSON if response_format is set, but we keep safety checks
        try:
            batch_results = json.loads(content)
        except json.JSONDecodeError:
            # Cleanup if markdown code blocks remain
            clean_text = content.replace("```json", "").replace("```", "").strip()
            batch_results = json.loads(clean_text)
            
        # --- 5. Remap & Standardize ---
        final_results = {}
        for simple_id, entities in batch_results.items():
            # Retrieve the ORIGINAL ID
            original_real_id = id_map.get(str(simple_id))
            
            if original_real_id:
                # Guard: skip if the LLM returned null/string instead of a list
                if not isinstance(entities, list):
                    final_results[original_real_id] = []
                else:
                    final_results[original_real_id] = _standardize_entities(entities)
            
        return final_results

    except Exception as e:
        print(f"   [!] OpenAI Batch Error: {e}")
        # Return empty so the pipeline continues
        return {}

def _parse_llm_json(result_text, event_id):
    """Helper to parse and standardize single-event JSON response"""
    clean_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
    clean_text = clean_text.replace("```json", "").replace("```", "").strip()
    
    try:
        entities = json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', clean_text, re.DOTALL)
        if match:
            try:
                entities = json.loads(match.group(0))
            except:
                return []
        else:
            return []
            
    return _standardize_entities(entities)

def _standardize_entities(entities):
    """Shared standardization logic (filtering publishers, fixing keys)"""
    # Guard against None or non-collection types
    if not entities or not isinstance(entities, (list, dict)):
        return []
    
    standardized = []
    
    # Handle dict wrapper {"entities": [...]} if present inside the list item (rare but possible)
    if isinstance(entities, dict):
         for key in entities:
            if isinstance(entities[key], list):
                entities = entities[key]; break
    
    if not isinstance(entities, list): return []

    for ent in entities:
        name = ent.get('name') or ent.get('entity_name')
        etype = ent.get('type') or ent.get('entity_type')
        role = ent.get('role')
        
        if isinstance(name, list): name = " ".join([str(n) for n in name])
        if isinstance(role, list): role = ", ".join([str(r) for r in role])
        
        if name and etype:
            name = str(name).strip()
            role = str(role).strip() if role else "Mentioned"
            etype_clean = 'Person' if 'person' in str(etype).lower() else 'Organization'
            
            if name.upper() in ['SHAB', 'SOGC', 'FOSC', 'KAB', 'KANT', 'AMTSBLATT']:
                continue

            standardized.append({
                'name': name,
                'type': etype_clean, 
                'role': role,
                'confidence': 'High (LLM)'
            })
            
    return standardized
