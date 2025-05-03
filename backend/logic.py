import json
import httpx
import time
import traceback
from typing import Tuple, AsyncGenerator, Optional, Dict, Any, List
import asyncio
from uuid import UUID, uuid4

# ייבוא מפתחות API ומשתנים מהקונפיגורציה
from config import CLAUDE_API_KEY, DEEPSEEK_API_KEY, supabase_client

# הגדרת משתנים גלובליים למודלים (נוכל לשפר את זה בהמשך, אולי דרך config)
# הערה: כרגע ערכים אלה קבועים. הפונקציה check_available_models לא הועברה כי היא תלויה ב-logger של הבוט
# וייתכן שתרצה דרך אחרת לקבוע את המודלים בסביבת Web.
ROUTER_MODEL = "claude-3-haiku-20240307" # נשתמש במודל מהיר וזול יותר לניתוב
PROMPT_ENGINEER_MODEL = "claude-3-haiku-20240307" # נשתמש במודל מהיר וזול יותר לשכתוב
CLAUDE_MODEL = "claude-3-opus-20240229" # המודל הראשי של קלוד
DEEPSEEK_MODEL = "deepseek-chat" # הגדרת מודל DeepSeek

# הגדרות הגבלת קצב (ללא שימוש כרגע, כי REQUEST_COUNTER היה גלובלי ברמת הבוט)
# MAX_REQUESTS_PER_MINUTE = 10
# REQUEST_COUNTER = 0
# LAST_REQUEST_RESET = 0

# --- Supabase Context Management Functions ---

async def create_user(auth_user_id: Optional[UUID] = None) -> Optional[UUID]:
    """
    Creates a new user in the 'users' table.
    Optionally links to a Supabase Auth user ID.
    Returns the new user's UUID or None on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    try:
        user_data = {}
        if auth_user_id:
            user_data['auth_user_id'] = str(auth_user_id)

        response = supabase_client.table('users').insert(user_data).execute()

        # Check response structure based on supabase-py v1+
        # Success if data is present
        if response.data and len(response.data) > 0:
            return UUID(response.data[0]['id'])
        else:
            # If execute() didn't raise an exception but no data, log it.
            print(f"Error creating user: No data returned and no exception raised.")
            return None
    except Exception as e:
        print(f"Exception in create_user: {e}")
        print(traceback.format_exc())
        return None

async def get_user_by_auth_id(auth_user_id: UUID) -> Optional[UUID]:
    """
    Finds a user's internal UUID based on their Supabase Auth ID.
    Returns the internal user UUID or None if not found or on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    try:
        response = supabase_client.table('users').select('id').eq('auth_user_id', str(auth_user_id)).limit(1).execute()

        # Check if data was returned
        if response.data and len(response.data) > 0:
            return UUID(response.data[0]['id'])
        else:
            # No user found or other issue (no data returned)
            return None # Not necessarily an error, just not found
    except Exception as e:
        print(f"Exception in get_user_by_auth_id: {e}")
        print(traceback.format_exc())
        return None

async def get_or_create_user_profile_context(user_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Retrieves the user's profile context. If it doesn't exist, creates an empty one.
    Returns the context data (dict) or None on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    try:
        # Try to get existing context
        response = supabase_client.table('user_profile_context').select('context_data').eq('user_id', str(user_id)).limit(1).execute()

        if response.data and len(response.data) > 0:
            return response.data[0].get('context_data', {})
        else:
             # Context not found, try creating one
            print(f"User profile context not found for user {user_id}, creating one.")
            initial_context = {}
            insert_response = supabase_client.table('user_profile_context').insert({
                'user_id': str(user_id),
                'context_data': initial_context
            }).execute()

            # Check insert response
            if insert_response.data and len(insert_response.data) > 0:
                return initial_context
            else:
                print(f"Error creating user profile context: No data returned from insert.")
                return None
    except Exception as e:
        # This will catch errors from both select and insert execute()
        print(f"Exception in get_or_create_user_profile_context: {e}")
        print(traceback.format_exc())
        return None

async def update_user_profile_context(user_id: UUID, context_data: Dict[str, Any]) -> bool:
    """
    Updates the user's profile context data.
    Returns True on success, False on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return False
    try:
        # The update method might not return data by default unless specified.
        # We rely on it *not* raising an exception for success.
        supabase_client.table('user_profile_context') \
            .update({'context_data': context_data, 'updated_at': 'now()'}) \
            .eq('user_id', str(user_id)) \
            .execute()
        return True # Assume success if no exception was raised
    except Exception as e:
        print(f"Exception in update_user_profile_context: {e}")
        print(traceback.format_exc())
        return False

async def create_conversation(user_id: UUID, title: Optional[str] = None) -> Optional[UUID]:
    """
    Creates a new conversation for a user and an initial empty context for it.
    Returns the new conversation UUID or None on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    conversation_id: Optional[UUID] = None
    try:
        # 1. Create conversation entry
        convo_data = {'user_id': str(user_id)}
        if title:
            convo_data['title'] = title

        print(f"Attempting to insert conversation for user {user_id}...")
        convo_response = supabase_client.table('conversations').insert(convo_data).execute()

        # Check if data was returned (indicates success)
        if convo_response.data and len(convo_response.data) > 0:
            conversation_id = UUID(convo_response.data[0]['id'])
            print(f"Conversation created with ID: {conversation_id}")
        else:
            print(f"Error: Conversation insert failed. No data returned.")
            return None

        # 2. Create initial conversation context
        initial_context = {}
        print(f"Attempting to insert context for conversation {conversation_id}...")
        context_response = supabase_client.table('conversation_context').insert({
            'conversation_id': str(conversation_id),
            'context_data': initial_context
        }).execute()

        # Check context creation success
        if not (context_response.data and len(context_response.data) > 0):
             print(f"Error: Conversation context insert failed for convo {conversation_id}. No data returned.")
             # Consider cleanup? For now, return None.
             return None

        print(f"Successfully created conversation and initial context for {conversation_id}")
        return conversation_id

    except Exception as e:
        print(f"Exception in create_conversation: {e}")
        print(traceback.format_exc())
        # If conversation was partially created, it might remain.
        return None

async def get_conversation_context(conversation_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Retrieves the context data for a specific conversation.
    Returns the context data (dict) or None if not found or on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    try:
        response = supabase_client.table('conversation_context').select('context_data').eq('conversation_id', str(conversation_id)).limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0].get('context_data', {})
        else:
            # Not found is not an error here, just return None
            print(f"Conversation context not found for {conversation_id}")
            return None
    except Exception as e:
        print(f"Exception in get_conversation_context: {e}")
        print(traceback.format_exc())
        return None

async def update_conversation_context(conversation_id: UUID, context_data: Dict[str, Any]) -> bool:
    """
    Updates the conversation's context data.
    Returns True on success, False on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return False
    try:
        # Rely on execute() not raising an exception for success.
        supabase_client.table('conversation_context') \
            .update({'context_data': context_data, 'updated_at': 'now()'}) \
            .eq('conversation_id', str(conversation_id)) \
            .execute()
        return True
    except Exception as e:
        print(f"Exception in update_conversation_context: {e}")
        print(traceback.format_exc())
        return False

async def add_message(conversation_id: UUID, role: str, content: str) -> bool:
    """
    Adds a message to a conversation.
    Role should be 'user' or 'assistant' (or 'system').
    Returns True on success, False on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return False
    if role not in ['user', 'assistant', 'system']:
         print(f"Error: Invalid role '{role}' specified for message.")
         return False
    try:
        # Check success by ensuring execute() doesn't raise an exception
        response = supabase_client.table('messages').insert({
            'conversation_id': str(conversation_id),
            'role': role,
            'content': content
        }).execute()
        # Check if data was returned (optional, insert might return data)
        if not (response.data and len(response.data) > 0):
            # This might not be an error, insert might not return data by default
            # print(f"Warning: Insert message did not return data for convo {conversation_id}")
            pass # Assuming success if no exception
        return True
    except Exception as e:
        print(f"Exception in add_message: {e}")
        print(traceback.format_exc())
        return False

async def get_messages(conversation_id: UUID, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves the last 'limit' messages for a conversation, ordered by creation time ascending.
    Returns a list of messages (dicts) or None on error.
    """
    if not supabase_client:
        print("Error: Supabase client not initialized.")
        return None
    try:
        response = supabase_client.table('messages') \
            .select('id, role, content, created_at') \
            .eq('conversation_id', str(conversation_id)) \
            .order('created_at', desc=False) \
            .limit(limit) \
            .execute()
        if response.data:
            return response.data
        else:
            # No messages found is not an error, return empty list
            return []
    except Exception as e:
        print(f"Exception in get_messages: {e}")
        print(traceback.format_exc())
        return None

# --- פונקציות עזר שהועברו ---

def get_model_description(model_name):
    """מחזיר תיאור מותאם למודל לפי שמו"""
    # הורדנו את האימוג'ים, כי הם לא תמיד מוצגים טוב ב-HTML/JSON
    if model_name.lower() == "claude":
        return "מודל המתמחה בניתוח טקסטים ויצירת תוכן מורכב"
    elif model_name.lower() == "deepseek":
        return "מודל המתמחה בקוד ובמשימות טכניות"
    else:
        return "מודל AI מתקדם"

def looks_like_code(text):
    """בדיקה אם טקסט נראה כמו קוד"""
    code_indicators = ["def ", "class ", "import ", "function", "var ", "let ", "const ", "<html", "public class", "if ", "for ", "#include"]
    return any(indicator in text for indicator in code_indicators)

def detect_programming_language(code):
    """זיהוי שפת תכנות לפי תוכן הקוד"""
    if "def " in code or "import " in code or "class " in code and ":" in code:
        return "python"
    elif "function" in code or "var " in code or "let " in code or "const " in code:
        return "javascript"
    elif "<html" in code or "<div" in code:
        return "html"
    elif "public class" in code or "private void" in code:
        return "java"
    elif "#include" in code:
        return "cpp"
    else:
        return "python"  # ברירת מחדל

# --- פונקציות ליבה שהועברו והותאמו ---

async def route_prompt(user_prompt: str) -> Tuple[str, str]:
    """
    Agent for routing - decides which model to send the prompt to.
    Returns: (model_name, reason - simplified to just model name for now)
    """
    if not CLAUDE_API_KEY:
        return "Error", "CLAUDE_API_KEY is not defined"

    # Simplified prompt asking only for the model name
    router_prompt = f"""
    You are a routing agent deciding between two AI models:
    1. Claude - Best for creative text, long content, complex reasoning.
    2. DeepSeek - Best for code, technical topics, concise questions.

    User prompt: "{user_prompt}"

    Based *only* on the user prompt, which model is more suitable? Respond with *only* the model name: "Claude" or "DeepSeek". Do not add any explanation or other text.
    """

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": ROUTER_MODEL,
                    "max_tokens": 10, # Reduced tokens for simple response
                    "messages": [{"role": "user", "content": router_prompt}]
                },
                timeout=15.0 # Reduced timeout for simple response
            )

            response.raise_for_status()

            router_response = response.json()
            model_choice_raw = router_response["content"][0]["text"].strip().replace('"', '') # Get text and remove quotes

            # Simple validation
            if "claude" in model_choice_raw.lower():
                model = "Claude"
            elif "deepseek" in model_choice_raw.lower():
                model = "DeepSeek"
            else:
                # If response is unexpected, default to Claude and log a warning
                print(f"Warning: Unexpected response from router: '{model_choice_raw}'. Defaulting to Claude.")
                model = "Claude"

            # Simplified reason (just the model name for now)
            reason = f"Router selected: {model}"

            # Keep the override logic for code-like prompts
            if looks_like_code(user_prompt) and model != "DeepSeek":
                print(f"Routing Conflict: Prompt looks like code, but router chose {model}. Overriding to DeepSeek.")
                model = "DeepSeek"
                reason = f"Router selected: {model} (Overridden due to code detection)"
            elif not looks_like_code(user_prompt) and model == "DeepSeek":
                 print(f"Routing Warning: Prompt doesn't look like code, but router chose DeepSeek.")

            return model, reason

        except httpx.RequestError as e:
            print(f"HTTP Error in routing agent: {e}")
            return "Error", f"Network error in routing agent: {e}"
        except Exception as e:
            print(f"General error in routing agent: {e}")
            print(traceback.format_exc())
            # Fallback routing
            if len(user_prompt) < 100 and any(kw in user_prompt.lower() for kw in ["code", "python", "javascript", "קוד", "תכנות"]):
                return "DeepSeek", "Fallback routing - code keywords detected"
            else:
                return "Claude", "Fallback routing - default"

async def engineer_prompt(user_prompt: str, model_choice: str) -> str:
    """
    סוכן שכתוב פרומפט - מתאים את הפרומפט למודל הנבחר
    """
    if not CLAUDE_API_KEY:
        return f"[שגיאה: CLAUDE_API_KEY אינו מוגדר] {user_prompt}"

    engineering_prompt = f"""
    אתה סוכן שכתוב פרומפט מומחה. התפקיד שלך הוא לקחת פרומפט מקורי של משתמש ולשכתב אותו כך שיהיה מיטבי עבור מודל ה-AI שנבחר.

    הפרומפט המקורי: "{user_prompt}"
    המודל שנבחר: {model_choice}

    אם נבחר Claude:
    - הוסף הנחיות מפורטות יותר לתשובה
    - הדגש את הצורך בתשובה מושקעת, מקיפה ומדויקת
    - התאם את הפרומפט ליכולות החשיבה וההנמקה המעמיקות של Claude

    אם נבחר DeepSeek:
    - שכתב את הפרומפט עם דגש על מבנה ברור ותמציתי
    - הוסף הנחיות ספציפיות לקוד או תוכן טכני אם רלוונטי
    - התאם את השאלה ליכולות התכנות והידע הטכני של DeepSeek

    החזר רק את הפרומפט המשוכתב, ללא הסברים או תוספות.
    """

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": PROMPT_ENGINEER_MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": engineering_prompt}]
                },
                timeout=30.0
            )

            if response.status_code != 200:
                 response.raise_for_status()

            engineer_response = response.json()
            return engineer_response["content"][0]["text"].strip()
    except httpx.RequestError as e:
            print(f"שגיאת HTTP בסוכן שכתוב: {e}")
            return f"[שגיאת רשת בשכתוב פרומפט] {user_prompt}"
    except Exception as e:
        print(f"שגיאה בשכתוב הפרומפט: {str(e)}")
        print(traceback.format_exc())
        # במקרה של שגיאה, נחזיר את הפרומפט המקורי עם הערה
        prefix = f"[שכתוב נכשל, פרומפט מקורי ל-{model_choice}]: "
        return prefix + user_prompt

async def stream_claude_response(client: httpx.AsyncClient, model: str, messages: List[Dict[str, str]], max_tokens: int) -> AsyncGenerator[str, None]:
    """Generator for streaming Claude API response."""
    # Ensure messages is actually a list of dicts
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        print(f"Error: stream_claude_response received invalid messages type: {type(messages)}")
        yield f"[STREAM_ERROR: Internal error - invalid message format passed to Claude streamer]"
        return

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages, # Ensure this is the list
        "stream": True
    }

    # Add anthropic specific headers (assuming client has base headers)
    headers = {
         "anthropic-version": "2023-06-01",
         "x-api-key": client.headers.get("x-api-key"), # Assuming key is in client headers now
         "content-type": "application/json"
         # "anthropic-beta": "messages-2023-12-15" # Might still be needed for some features
    }
    try:
        # --- Log the exact payload being sent ---
        try:
            payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
            print(f"--- Sending Payload to Claude API ---\n{payload_str[:1000]}...\n-------------------------------------")
        except Exception as log_e:
            print(f"Error logging payload: {log_e}")
        # -----------------------------------------

        # Pass headers explicitly to the stream call
        async with client.stream("POST", "https://api.anthropic.com/v1/messages", json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"): # Process only data lines for SSE
                    try:
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]": # OpenAI style done message, Claude might use something else or just close
                            break
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta":
                             yield data["delta"]["text"]
                        elif data.get("type") == "message_stop":
                             break # Stop when Claude signals the end
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from stream line: {line}")
                        continue
                    except Exception as e:
                        print(f"Error processing stream line: {line} - {e}")
                        continue
    except httpx.HTTPStatusError as e:
        error_content = await e.response.aread()
        print(f"HTTP Error during Claude stream: {e.response.status_code} - {error_content}")
        # Decode bytes to string if possible for cleaner logging/error message
        error_text = error_content.decode('utf-8', errors='ignore') if isinstance(error_content, bytes) else str(error_content)
        yield f"[STREAM_ERROR: HTTP {e.response.status_code} - {error_text[:200]}]" # Truncate long errors
    except Exception as e:
        print(f"General Error during Claude stream: {e}")
        print(traceback.format_exc())
        yield "[STREAM_ERROR: General Error]"


async def stream_deepseek_response(client: httpx.AsyncClient, model: str, messages: List[Dict[str, str]], max_tokens: int) -> AsyncGenerator[str, None]:
    """Generator for streaming DeepSeek API response."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True
    }
    try:
        print("[DEEPSEEK_STREAM] Initiating POST request to DeepSeek API.")
        async with client.stream("POST", "https://api.deepseek.com/v1/chat/completions", json=payload) as response:
            print(f"[DEEPSEEK_STREAM] Received response status: {response.status_code}")
            response.raise_for_status() 
            async for line in response.aiter_lines():
                print(f"[DEEPSEEK_STREAM_RAW] Received line: {line[:500]}") # Log raw line (truncated)
                if not line or not line.startswith("data:"): 
                    continue
                
                data_str = line[len("data:"):].strip()
                print(f"[DEEPSEEK_STREAM_DATA] Data string: {data_str}") # Log data string
                
                if data_str == "[DONE]": 
                    print("[DEEPSEEK_STREAM_DONE] Received [DONE] signal.")
                    break 
                
                try:
                    data = json.loads(data_str)
                    print(f"[DEEPSEEK_STREAM_JSON] Parsed JSON: {json.dumps(data, indent=2)[:500]}") # Log parsed JSON (truncated)
                    if data.get("choices") and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        finish_reason = choice.get("finish_reason")
                        print(f"[DEEPSEEK_STREAM_EXTRACT] Content: '{content}', Finish Reason: {finish_reason}") # Log extraction results
                        
                        if content:
                            print("[DEEPSEEK_STREAM_YIELD] Yielding content chunk...") # Log before yield
                            yield content
                            
                        if finish_reason:
                            print(f"[DEEPSEEK_STREAM_FINISH] Breaking loop due to finish_reason: {finish_reason}")
                            break 
                            
                except json.JSONDecodeError:
                    print(f"[DEEPSEEK_STREAM_WARN] Could not decode JSON: {line}")
                    continue 
                except Exception as e:
                    print(f"[DEEPSEEK_STREAM_ERROR] Error processing line: {line} - {e}")
                    continue 
                    
    except httpx.HTTPStatusError as e:
        error_content = b"Unknown error"
        try:
             error_content = await e.response.aread()
        except Exception as read_err:
             print(f"Could not read error response body: {read_err}")
        print(f"HTTP Error during DeepSeek stream: {e.response.status_code} - {error_content}")
        error_text = error_content.decode('utf-8', errors='ignore') if isinstance(error_content, bytes) else str(error_content)
        yield f"[STREAM_ERROR: HTTP {e.response.status_code} - {error_text[:200]}]"
    except Exception as e:
        print(f"General Error during DeepSeek stream: {e}")
        print(traceback.format_exc())
        yield "[STREAM_ERROR: General Error]"

async def call_ai_model_stream(messages_list: List[Dict[str, str]], model_choice: str) -> AsyncGenerator[str, None]:
    """
    Calls the appropriate AI model (Claude or DeepSeek) in streaming mode.
    Yields chunks of the response or an error string starting with [STREAM_ERROR:].
    """
    max_tokens = 4000 # Maximum tokens for the response

    # --- Validate and Clean Messages FIRST ---
    try:
        cleaned_messages_list = validate_and_clean_message_list(messages_list)
        if not cleaned_messages_list: # Should not happen if validation works, but check anyway
             yield "[STREAM_ERROR: Message list became empty after cleaning]"
             return
    except ValueError as val_err:
         print(f"Message list validation failed: {val_err}")
         yield f"[STREAM_ERROR: Invalid message structure - {val_err}]"
         return
    except Exception as clean_err:
         print(f"Unexpected error during message cleaning: {clean_err}")
         traceback.print_exc()
         yield f"[STREAM_ERROR: Internal error during message preparation]"
         return

    # --- Proceed with cleaned list ---
    messages_to_send = cleaned_messages_list

    # Simple routing logic (can be expanded)
    if model_choice.lower() == 'claude':
        if not CLAUDE_API_KEY:
            yield "[STREAM_ERROR: Claude API key not configured]"
            return
        target_model = CLAUDE_MODEL
        api_key = CLAUDE_API_KEY
        stream_func = stream_claude_response
        print(f"Attempting to call Claude model ({target_model}) with {len(messages_to_send)} messages.")
        print(f"Using Claude API Key: {api_key[:5]}...{api_key[-4:]}") # Log partial key
        # Log the messages being sent (be careful with sensitive data in production)
        print("Messages sent to Claude:")
        for i, msg in enumerate(messages_to_send):
            print(f"  [{i}] {msg['role']}: {str(msg['content'])[:150]}...") # Log truncated content

    elif model_choice.lower() == 'deepseek':
        if not DEEPSEEK_API_KEY:
            yield "[STREAM_ERROR: DeepSeek API key not configured]"
            return
        target_model = DEEPSEEK_MODEL
        api_key = DEEPSEEK_API_KEY # Using the imported DeepSeek key
        stream_func = stream_deepseek_response
        print(f"Attempting to call DeepSeek model ({target_model}) with {len(messages_to_send)} messages.")
        print(f"Using DeepSeek API Key: {api_key[:5]}...{api_key[-4:]}") # Log partial key
        # Log the messages being sent
        print("Messages sent to DeepSeek:")
        for i, msg in enumerate(messages_to_send):
             print(f"  [{i}] {msg['role']}: {str(msg['content'])[:150]}...") # Log truncated content
    else:
        yield f"[STREAM_ERROR: Unknown model choice '{model_choice}']"
        return

    # --- Set up headers based on model choice ---
    headers = {}
    if model_choice.lower() == 'claude':
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    elif model_choice.lower() == 'deepseek':
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    # Use httpx.AsyncClient for making requests, now with correct headers
    async with httpx.AsyncClient(timeout=120.0, headers=headers) as client:
        try:
            print(f"Initiating stream with {model_choice}...")
            # Pass the client WITHOUT adding headers again inside stream_func
            async for chunk in stream_func(client, target_model, messages_to_send, max_tokens):
                yield str(chunk)
            print(f"Stream finished successfully with {model_choice}.")

        except httpx.HTTPStatusError as status_err:
            error_body = await status_err.response.aread()
            error_details = f"HTTP Status Error calling {model_choice}: {status_err.response.status_code} - {error_body.decode()[:500]}"
            print(error_details) # Log detailed error
            yield f"[STREAM_ERROR: {error_details}]"
        except httpx.RequestError as req_err:
            error_details = f"Request Error calling {model_choice}: {req_err}"
            print(error_details) # Log detailed error
            yield f"[STREAM_ERROR: {error_details}]"
        except Exception as e:
            error_details = f"Unexpected error during {model_choice} stream: {e}"
            print(error_details) # Log detailed error
            traceback.print_exc()
            yield f"[STREAM_ERROR: {error_details}]"

# --- Helper function to validate/clean messages for API ---
def validate_and_clean_message_list(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Validates message list format for Claude/DeepSeek APIs.
    Ensures alternating user/assistant roles (after optional system prompt).
    Ensures 'role' and 'content' exist and are strings.
    Returns a cleaned list or raises ValueError if unfixable issues are found.
    """
    cleaned_messages = []
    last_role = None

    if not messages:
        raise ValueError("Message list cannot be empty")

    # Handle optional system prompt first
    if messages[0].get('role') == 'system':
        content = messages[0].get('content')
        if isinstance(content, str):
            cleaned_messages.append({'role': 'system', 'content': content})
            messages = messages[1:] # Process the rest
            last_role = 'system' # For alternating check
        else:
            raise ValueError("System prompt content must be a string.")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message at index {i} (after system) is not a dictionary: {msg}")

        role = msg.get('role')
        content = msg.get('content')

        if role not in ['user', 'assistant']:
            raise ValueError(f"Invalid role '{role}' found at index {i} (after system).")

        if not isinstance(content, str):
             # Try to convert if possible, otherwise raise error
             try:
                 content = str(content)
             except Exception:
                  raise ValueError(f"Content for role '{role}' at index {i} (after system) is not a string and cannot be converted: {type(content)}")

        # Check for alternating roles
        if last_role == role:
            # If consecutive roles, we might need to merge or skip, but for now raise error for Claude
            # For simplicity, let's assume Claude API requires strict alternation
             raise ValueError(f"Consecutive messages found with the same role '{role}' at index {i} (after system). API requires alternating roles.")

        cleaned_messages.append({'role': role, 'content': content})
        last_role = role

    # Final check: Last message role (if not just system prompt) should typically be 'user' for Claude
    if cleaned_messages and cleaned_messages[-1]['role'] == 'assistant':
         print("Warning: The last message sent to the AI is from the assistant. This might cause issues.")
         # Depending on API strictness, this might be an error. For now, just a warning.

    return cleaned_messages 