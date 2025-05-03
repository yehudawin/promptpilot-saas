import os
from dotenv import load_dotenv
from supabase import create_client, Client

# טעינת משתני סביבה מקובץ .env
load_dotenv()

# קריאת מפתחות ה-API ממשתני הסביבה
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# ניתן להוסיף כאן מפתחות API נוספים בעתיד

# --- Supabase Configuration Start ---
# Use uppercase names consistent with how they are likely imported/used elsewhere
SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY: str | None = os.environ.get("SUPABASE_ANON_KEY")
supabase_client: Client | None = None # Initialize client as None

if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
else:
    print("Warning: SUPABASE_URL or SUPABASE_ANON_KEY environment variables are not set.")
# --- Supabase Configuration End ---

# בדיקה אם המפתחות הוגדרו
if not CLAUDE_API_KEY:
    print("אזהרה: משתנה הסביבה CLAUDE_API_KEY אינו מוגדר.")
if not DEEPSEEK_API_KEY:
    print("אזהרה: משתנה הסביבה DEEPSEEK_API_KEY אינו מוגדר.") 