import uvicorn
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import traceback
import json
import asyncio
from uuid import UUID, uuid4
from typing import Optional, List, Dict
import time # Add time import

# ייבוא הלוגיקה מהמודול הנפרד
from backend.logic import (
    route_prompt, engineer_prompt, call_ai_model_stream, get_model_description,
    create_user, get_user_by_auth_id, get_or_create_user_profile_context, update_user_profile_context,
    create_conversation, get_conversation_context, update_conversation_context,
    add_message, get_messages
)
# Import supabase client to check initialization
from backend.config import supabase_client, SUPABASE_URL, SUPABASE_ANON_KEY # Import keys too

# הגדרת אפליקציית FastAPI
app = FastAPI()

# --- Default User Setup (Simplified) ---
DEFAULT_USER_ID: Optional[UUID] = None

@app.on_event("startup")
async def startup_event():
    global DEFAULT_USER_ID
    print("Running startup event...")
    if not supabase_client:
        print("ERROR: Supabase client not available during startup.")
        # Decide how to handle this - maybe raise an error or prevent app start?
        # For now, we'll allow startup but context features will fail.
        return

    # Let's assume a single default user for now.
    # We could store this ID in a file or env var, but for simplicity:
    # Check if a user exists (e.g., based on a known marker in profile or just grab first one)
    # This is very basic - a real app needs proper user management/auth.
    try:
        # A better approach would be to fetch based on a specific criteria if possible
        # For now, just try creating one and see if it works or assume one exists
        # We need a way to reliably get the *same* default user ID each time.
        # Let's try creating one with a specific known UUID if it doesn't exist.
        # THIS IS NOT SECURE OR ROBUST - Placeholder only.
        # A fixed UUID for the default user
        known_default_auth_id = UUID("00000000-0000-0000-0000-000000000001") # Example placeholder
        print(f"Attempting to find or create default user with auth ID: {known_default_auth_id}")
        
        # Try to get user by the known placeholder auth ID
        existing_user_id = await get_user_by_auth_id(known_default_auth_id)

        if existing_user_id:
            DEFAULT_USER_ID = existing_user_id
            print(f"Found existing default user with internal ID: {DEFAULT_USER_ID}")
        else:
            print("Default user not found, attempting to create...")
            # Pass the known auth_id when creating
            new_user_id = await create_user(auth_user_id=known_default_auth_id)
            if new_user_id:
                DEFAULT_USER_ID = new_user_id
                print(f"Created new default user with internal ID: {DEFAULT_USER_ID}")
                # Optionally create initial profile context here
                await get_or_create_user_profile_context(DEFAULT_USER_ID)
            else:
                print("ERROR: Failed to create default user during startup.")
                # Handle error - maybe raise exception?

    except Exception as e:
        print(f"ERROR during default user setup: {e}")
        traceback.print_exc()
        # Application might not function correctly without a user ID.

# הגדרת מיקום לקבצים סטטיים (CSS, JS, תמונות - אם יהיו)
# נניח שהם יהיו בתיקיית 'static' שצריך ליצור
# אם ה-JS/CSS משולבים ב-HTML, זה פחות קריטי כרגע
# נכון לעכשיו, אין לנו תיקיית static, אז נגיש קבצים ישירות
# app.mount(\"/static\", StaticFiles(directory=\"static\"), name=\"static\")

# הגדרת תבניות HTML (אם נרצה להעביר נתונים ל-HTML מהשרת)
# כרגע נראה שקבצי ה-HTML הם סטטיים בעיקרם
# templates = Jinja2Templates(directory=\".\")

# מודל קלט עבור הבקשה ל-API
class PromptRequest(BaseModel):
    prompt: str

# --- API Endpoint for Frontend Configuration ---

@app.get("/api/config", response_class=JSONResponse)
async def get_frontend_config():
    """
    Provides necessary configuration (like Supabase keys) to the frontend.
    Reads directly from imported config variables.
    """
    print("Serving /api/config request...")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Error: Supabase URL or Anon Key missing in configuration.")
        raise HTTPException(status_code=500, detail="Server configuration error: Supabase keys not found.")
    
    config_data = {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY
    }
    print(f"Returning config: URL={config_data['supabase_url']}, Key=***") # Don't log the full key
    return JSONResponse(content=config_data)

# --- SSE Streaming Endpoint ---

# Define constants
MESSAGE_HISTORY_LIMIT = 10 # How many past messages to include (excluding new user message)
DEFAULT_TITLE_LENGTH = 40
MAX_CONTEXT_CHARS = 500 # Limit context string length in system prompt

# Modify stream_generator signature and logic
async def stream_generator(user_prompt: str, user_id: UUID, conversation_id: UUID):
    """
    Generates Server-Sent Events including context retrieval/update and AI response.
    Uses message history for the main AI call.
    """
    full_ai_response = ""
    model_choice = "Unknown"
    final_messages_list: List[Dict[str, str]] = [] # Initialize

    try:
        # --- Context & History Retrieval ---
        print(f"Processing stream for user: {user_id}, conversation: {conversation_id}")
        profile_context = await get_or_create_user_profile_context(user_id)
        convo_context = await get_conversation_context(conversation_id)
        # 1. Fetch history *before* adding the new user message
        message_history_dicts = await get_messages(conversation_id, limit=MESSAGE_HISTORY_LIMIT)

        if profile_context is None or convo_context is None or message_history_dicts is None:
             error_data = json.dumps({"error": "Failed to retrieve context or message history"})
             yield f"event: error\ndata: {error_data}\n\n"
             return

        # --- Prepare Message List for AI --- (Do this BEFORE saving the new user message)
        # 1. Start with System Prompt (Optional but Recommended)
        system_prompt_content = "You are a helpful assistant. "
        # Add user preferences (limited length)
        if profile_context:
            pref_str = json.dumps(profile_context)
            system_prompt_content += f"User preferences: {pref_str[:MAX_CONTEXT_CHARS]}{'...' if len(pref_str) > MAX_CONTEXT_CHARS else ''}. "
        # Add conversation context (limited length)
        if convo_context:
            convo_ctx_str = json.dumps(convo_context)
            system_prompt_content += f"Conversation context: {convo_ctx_str[:MAX_CONTEXT_CHARS]}{'...' if len(convo_ctx_str) > MAX_CONTEXT_CHARS else ''}."

        # Only add system prompt if it has more than the base text
        if len(system_prompt_content) > len("You are a helpful assistant. "):
             final_messages_list.append({"role": "system", "content": system_prompt_content.strip()})

        # 2. Add historical messages (convert from DB format if needed)
        # Add validation for alternating roles
        last_added_role = final_messages_list[-1]["role"] if final_messages_list else None

        for msg in message_history_dicts:
             # Ensure only role and content are included and role alternates
             if 'role' in msg and 'content' in msg:
                 current_role = msg['role']
                 # Skip if current role is same as the last one added
                 if current_role == last_added_role:
                     print(f"Skipping message with role '{current_role}' to maintain alternation.")
                     continue
                 # Add the message and update last_added_role
                 final_messages_list.append({"role": current_role, "content": msg['content']})
                 last_added_role = current_role

        # 3. Add the NEW user message to the list being sent to the AI
        # Ensure the new user message doesn't immediately follow another user message
        if last_added_role == 'user':
             # This case is problematic. Decide how to handle.
             # Option 1: Skip adding the new user message (might lose context)
             # Option 2: Try to insert a placeholder assistant message (might confuse AI)
             # Option 3: Log an error and potentially stop (safest but disruptive)
             print(f"Error: New user message would follow another user message. History state: {last_added_role}")
             # For now, let's log and *not* add the user message to prevent API error,
             # though this isn't ideal as the user's input is lost for this turn.
             error_data = json.dumps({"error": "Cannot process request: conversation history ended with user message."})
             yield f"event: error\\ndata: {error_data}\\n\\n"
             return # Stop the stream
        else:
            # Only add user message if the last message was assistant or system
            final_messages_list.append({"role": "user", "content": user_prompt})
            last_added_role = 'user' # Update last role after adding


        # --- Save User Message to DB NOW --- (After constructing the list for the current call)
        save_user_success = await add_message(conversation_id, 'user', user_prompt)
        if not save_user_success:
            print(f"Warning: Failed to save user message for conversation {conversation_id}")
            warning_payload = {"message": "Warning: Could not save user message to history."}
            yield f"event: warning\ndata: {json.dumps(warning_payload)}\n\n"
            # Note: We continue even if saving failed, as the message is in the list sent to AI

        print(f"Constructed message list for AI (length {len(final_messages_list)}):")
        for i, msg in enumerate(final_messages_list):
             print(f"  [{i}] {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")


        # --- Routing & Engineering (Simplified - based on *last user prompt* only) ---
        print("Routing based on last user prompt...")
        # Pass only the last user prompt for routing decision
        model_choice, reason = await route_prompt(user_prompt)
        if model_choice == "Error":
            error_data = json.dumps({"error": f"Routing error: {reason}"})
            yield f"event: error\ndata: {error_data}\n\n"
            return

        metadata = { # Metadata sent back to client
            "selected_model": model_choice,
            "model_description": get_model_description(model_choice),
            "reason": reason,
            "conversation_id": str(conversation_id)
        }
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        print("Engineering prompt based on last user prompt...")
        # Engineer only the last user prompt
        engineered_last_prompt = await engineer_prompt(user_prompt, model_choice)
        yield f"event: engineered_prompt\ndata: {json.dumps({'engineered_prompt': engineered_last_prompt})}\n\n"

        # Decide whether to use the engineered *last* prompt or the original *last* prompt
        # Note: The history *before* this last prompt remains unchanged in final_messages_list
        if engineered_last_prompt.startswith(("[שגיאה:", "[שכתוב נכשל,", "[שגיאת רשת בשכתוב פרומפט]")):
            print(f"Prompt engineering issue for last prompt: {engineered_last_prompt}")
            # If engineering failed, keep the original user prompt as the last message
            warning_payload = {"message": "Prompt engineering failed for the last message."}
            yield f"event: warning\ndata: {json.dumps(warning_payload)}\n\n"
        else:
            # If engineering succeeded, *replace* the content of the last message
            if final_messages_list and final_messages_list[-1]["role"] == "user":
                 print("Applying engineered prompt to the last user message.")
                 final_messages_list[-1]["content"] = engineered_last_prompt
            else:
                 print("Warning: Could not apply engineered prompt - last message was not 'user'.")


        # --- Call AI Model with Full Message List ---
        print(f"Calling model {model_choice} with {len(final_messages_list)} messages.")
        async for chunk in call_ai_model_stream(final_messages_list, model_choice): # Pass the list
            if chunk.startswith("[STREAM_ERROR:"):
                error_data = json.dumps({"error": chunk})
                yield f"event: error\ndata: {error_data}\n\n"
                # Optional: save error context update
                # await update_conversation_context(conversation_id, {'last_status': 'error', 'error_details': chunk})
                return # Stop stream
            else:
                full_ai_response += chunk
                chunk_data = json.dumps({"text": chunk})
                yield f"event: message\ndata: {chunk_data}\n\n"
            await asyncio.sleep(0.01)

        # --- Save Full Assistant Message ---
        if full_ai_response:
             print(f"Saving assistant response for convo {conversation_id}")
             save_assistant_success = await add_message(conversation_id, 'assistant', full_ai_response)
             if not save_assistant_success:
                 print(f"Warning: Failed to save assistant message for conversation {conversation_id}")
                 warning_payload = {"message": "Warning: Could not save assistant response to history."}
                 yield f"event: warning\ndata: {json.dumps(warning_payload)}\n\n"

        # --- Update Conversation Context ---
        print(f"Updating context for convo {conversation_id}")
        new_convo_context = convo_context.copy()
        new_convo_context['last_interaction_at'] = time.time()
        new_convo_context['last_model_used'] = model_choice
        new_convo_context['last_status'] = 'success'
        # TODO: Add summarization logic
        update_ctx_success = await update_conversation_context(conversation_id, new_convo_context)
        if not update_ctx_success:
            print(f"Warning: Failed to update conversation context for {conversation_id}")
            warning_payload = {"message": "Warning: Could not update conversation context."}
            yield f"event: warning\ndata: {json.dumps(warning_payload)}\n\n"

        yield f"event: end\ndata: End of stream\n\n"

    except HTTPException as http_exc:
        error_data = json.dumps({"error": http_exc.detail, "status_code": http_exc.status_code})
        yield f"event: error\ndata: {error_data}\n\n"
    except Exception as e:
        print(f"Unhandled error in stream_generator for convo {conversation_id}: {e}")
        traceback.print_exc()
        error_details = f"Internal server error: {str(e)}"
        error_data = json.dumps({"error": error_details, "model_used": model_choice})
        yield f"event: error\ndata: {error_data}\n\n"
        # Try to update context with error status if possible
        try:
            # Need a valid convo_context object even in failure case if possible
            ctx_to_update = await get_conversation_context(conversation_id) if 'convo_context' not in locals() or convo_context is None else convo_context
            if ctx_to_update is not None:
                ctx_to_update['last_status'] = 'error'
                ctx_to_update['error_details'] = error_details[:500] # Limit error length
                await update_conversation_context(conversation_id, ctx_to_update)
        except Exception as context_update_e:
            print(f"Failed to update context with error details: {context_update_e}")

# Modify the endpoint to accept conversation_id and handle creation
@app.get("/api/process_stream")
async def process_prompt_stream_endpoint(
    prompt: str = Query(...),
    conversation_id: Optional[str] = Query(None) # Make conversation_id optional
):
    print(">>> process_prompt_stream_endpoint ENTERED") # Add entry log
    user_prompt = prompt
    if not user_prompt:
        return JSONResponse(status_code=400, content={"error": "Prompt cannot be empty"})

    if DEFAULT_USER_ID is None:
         print("Error: Default user ID not available.")
         # Return JSONResponse instead of raising HTTPException for client handling
         return JSONResponse(status_code=503, content={"error": "Service unavailable: User initialization failed."})

    current_user_id = DEFAULT_USER_ID # Use the global default user ID
    current_conversation_id: Optional[UUID] = None

    if conversation_id:
        try:
            current_conversation_id = UUID(conversation_id)
            # Optional: Verify conversation belongs to user if implementing multi-user
            # convo_details = await get_conversation_details(current_conversation_id) # Needs implementation
            # if not convo_details or convo_details['user_id'] != current_user_id:
            #     return JSONResponse(status_code=403, content={"error": "Conversation access denied"})
            print(f"Using existing conversation ID: {current_conversation_id}")
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid conversation_id format"})
    else:
        # Create a new conversation
        print("No conversation_id provided, creating a new one...")
        # Generate a default title from the first few words of the prompt
        default_title = user_prompt[:DEFAULT_TITLE_LENGTH] + ('...' if len(user_prompt) > DEFAULT_TITLE_LENGTH else '')
        new_convo_id = await create_conversation(user_id=current_user_id, title=default_title)
        if not new_convo_id:
            # Handle creation failure
            print("Error: Failed to create new conversation in database.")
            return JSONResponse(status_code=500, content={"error": "Failed to create new conversation"})
        current_conversation_id = new_convo_id
        print(f"Created new conversation with ID: {current_conversation_id}")

    # Final check if we have a conversation ID
    if not current_conversation_id:
         print("Error: Failed to obtain a conversation ID.")
         return 
    JSONResponse(status_code=500, content={"error": "Failed to obtain conversation ID"})

    return StreamingResponse(
        stream_generator(user_prompt=user_prompt, user_id=current_user_id, conversation_id=current_conversation_id),
        media_type="text/event-stream"
    )

# --- Endpoint ישן (ללא סטרימינג) - נשאיר אותו בינתיים כגיבוי או לדיבאגינג ---
@app.post("/api/process")
async def process_prompt_endpoint(request: PromptRequest):
    user_prompt = request.prompt
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # 1. ניתוב הפרומפט
        model_choice, reason = await route_prompt(user_prompt)
        if model_choice == "Error": # בדיקת שגיאה מהניתוב
            raise HTTPException(status_code=500, detail=f"Routing error: {reason}")

        # 2. שיפור הפרומפט
        engineered_prompt = await engineer_prompt(user_prompt, model_choice)
        final_prompt_to_use = engineered_prompt # השתמש בפרומפט המשופר כברירת מחדל
        if engineered_prompt.startswith("[שגיאה:") or engineered_prompt.startswith("[שכתוב נכשל,") or engineered_prompt.startswith("[שגיאת רשת בשכתוב פרומפט]"):
            print(f"Prompt engineering issue: {engineered_prompt}")
            # נמשיך עם הפרומפט המקורי במקרה של כשלון בשכתוב
            final_prompt_to_use = user_prompt

        # 3. קריאה למודל ה-AI
        model_response = await call_ai_model(final_prompt_to_use, model_choice)
        # בדיקת שגיאה מהמודל
        if model_response.startswith("שגיאה:") or model_response.startswith("אירעה שגיאה") or model_response.startswith("[שגיאה:"):
            raise HTTPException(status_code=500, detail=f"AI model error: {model_response}")

        # החזרת התוצאות ל-Frontend
        return JSONResponse(content={
            "selected_model": model_choice,
            "model_description": get_model_description(model_choice),
            "reason": reason,
            "engineered_prompt": engineered_prompt, # נחזיר גם את המשופר
            "final_response": model_response
        })

    except HTTPException as http_exc: # לכדה מפורשת של שגיאות HTTP
        raise http_exc
    except Exception as e:
        print(f"Unhandled error in /api/process: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- הגשת קבצי HTML סטטיים ---
# נוודא ששמות הקבצים מדויקים, כולל סיומת

# נתיב הבסיס של הקבצים הסטטיים
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_HTML_FILES = [
    "Landing_Page.HTML",
    "Chat_page.html",
    "Setting_page.html",
    "Help&Support_page.html",
    "Admin_Dashboard.html"
]

@app.get("/{page_name}", response_class=HTMLResponse)
async def serve_html_page(page_name: str):
    # נוודא שהסיומת היא .html או .HTML
    if not page_name.lower().endswith(('.html', '.htm')):
        # אפשר להניח שזו בקשה לקובץ סטטי אחר או ל-API - נחזיר 404
        raise HTTPException(status_code=404, detail="Not an HTML file request")

    # נחפש את הקובץ המדויק או ברישיות שונה
    target_file = None
    for allowed_file in ALLOWED_HTML_FILES:
        if page_name.lower() == allowed_file.lower():
            # נבדוק אם הקובץ קיים בנתיב המלא
            potential_path = os.path.join(BASE_DIR, allowed_file)
            if os.path.isfile(potential_path):
                target_file = potential_path
                break

    if not target_file:
        raise HTTPException(status_code=404, detail=f"HTML file '{page_name}' not found or not allowed.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        print(f"Error reading file {target_file}: {e}")
        raise HTTPException(status_code=500, detail="Error reading HTML file")

# ניתוב עבור דף הבית ('/') שיפנה ל-Landing Page
@app.get("/", response_class=HTMLResponse)
async def serve_root_page(): # שינוי שם הפונקציה למניעת התנגשות
    # נגיש את Landing_Page.HTML
    landing_page_path = os.path.join(BASE_DIR, "Landing_Page.HTML")
    if not os.path.isfile(landing_page_path):
        # אולי זה Landing_Page.html? נבדוק
        landing_page_path_lower = os.path.join(BASE_DIR, "Landing_Page.html")
        if os.path.isfile(landing_page_path_lower):
            landing_page_path = landing_page_path_lower
        else:
            raise HTTPException(status_code=404, detail="Landing page file (Landing_Page.HTML or Landing_Page.html) not found.")

    try:
        with open(landing_page_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        print(f"Error reading landing page file {landing_page_path}: {e}")
        raise HTTPException(status_code=500, detail="Error reading landing page file")


# הרצת השרת (רק אם הקובץ מורץ ישירות)
if __name__ == "__main__":
    # מומלץ להשתמש ב-uvicorn מהטרמינל ולא כאן
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("להרצת השרת, השתמש בפקודה:")
    print("uvicorn main:app --reload --host 0.0.0.0 --port 8000") 