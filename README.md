# Prompt Pilot Dassy

A SaaS application for intelligent prompt routing and AI model selection, featuring a web interface and backend powered by FastAPI and Supabase.

## Overview

Prompt Pilot Dassy is designed to intelligently handle user prompts by:
1.  **Routing:** Selecting the most suitable AI model (e.g., Claude, DeepSeek) based on the prompt content.
2.  **Prompt Engineering:** Refining the user's prompt for optimal performance with the chosen model.
3.  **AI Interaction:** Calling the selected AI model and streaming the response back to the user.
4.  **Context Management:** Utilizing user profiles and conversation history stored in Supabase to provide contextual responses.

The application uses a FastAPI backend with Server-Sent Events (SSE) for real-time communication and a simple web frontend for user interaction.

## Features

-   FastAPI Backend with asynchronous operations.
-   Real-time response streaming using Server-Sent Events (SSE).
-   Intelligent prompt routing logic (`logic.py`).
-   Prompt engineering capabilities (`logic.py`).
-   Integration with multiple AI models (configurable in `logic.py`).
-   User profile and conversation context management via Supabase (`logic.py`, `config.py`).
-   Web interface (HTML/JS likely in the root or a dedicated frontend directory).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set up Supabase:**
    *   Create a Supabase project.
    *   Obtain your Supabase Project URL and Anon Key.
    *   Set up the required database tables (see `logic.py` for expected tables like `users`, `user_profiles`, `conversations`, `messages`). You might need to run SQL scripts or use Supabase Studio.

3.  **Configure Environment Variables:**
    *   The application expects Supabase credentials and API keys for the AI models. It's strongly recommended to use environment variables. Create a `.env` file in the `backend` directory or set system environment variables:
        ```dotenv
        # backend/.env
        SUPABASE_URL=your_supabase_project_url
        SUPABASE_ANON_KEY=your_supabase_anon_key
        ANTHROPIC_API_KEY=your_claude_api_key
        DEEPSEEK_API_KEY=your_deepseek_api_key
        # Add other necessary keys
        ```
    *   The application loads these in `config.py`.

4.  **Install Python dependencies:**
    *   Navigate to the `backend` directory.
    *   Create and activate a virtual environment (recommended):
        ```bash
        python -m venv venv
        # On Windows:
        venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    *   Install required packages:
        ```bash
        pip install -r requirements.txt
        ```
        *(Ensure `requirements.txt` exists in the `backend` directory and is up-to-date).*

## Running the Application

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reload on code changes (useful for development).
    *   `--host 0.0.0.0`: Makes the server accessible on your local network.
    *   `--port 8000`: Specifies the port to run on.

3.  **Access the application:**
    *   Open your web browser and navigate to `http://localhost:8000` or `http://<your-local-ip>:8000`.

## How it Works

1.  The user interacts with the web frontend, entering prompts.
2.  The frontend sends the prompt (and conversation ID, if applicable) to the FastAPI backend (`/api/process_stream`).
3.  The backend retrieves user and conversation context from Supabase.
4.  The `route_prompt` function selects the best AI model.
5.  The `engineer_prompt` function potentially modifies the prompt.
6.  The `call_ai_model_stream` function sends the processed prompt (along with history) to the chosen AI model's API.
7.  The backend streams the AI's response back to the frontend using SSE.
8.  User and assistant messages are saved to the Supabase database.

## Security

-   **NEVER** commit your API keys or Supabase credentials directly into your code or Git repository.
-   Use environment variables (`.env` file loaded by `python-dotenv` or system variables) to manage sensitive keys securely. Ensure your `.env` file is listed in your `.gitignore`.

## Development

-   The core logic resides in `backend/logic.py`.
-   API endpoints are defined in `backend/main.py`.
-   Database interactions and configurations are in `backend/config.py`.
-   The frontend files (HTML, CSS, JS) are likely served directly by FastAPI or located in a separate frontend directory. 