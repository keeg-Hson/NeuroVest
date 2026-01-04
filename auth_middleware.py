"""
Simple API Key Authentication Middleware
"""
import streamlit as st
import secrets
from core.data_manager_postgres import DataManager

class AuthManager:
    """Manage user authentication via API keys"""

    @staticmethod
    def generate_api_key():
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)

    @staticmethod
    def create_user(username=None):
        """Create a new user with API key"""
        dm = DataManager()
        api_key = AuthManager.generate_api_key()

        with dm.engine.begin() as conn:
            result = conn.execute(
                """
                INSERT INTO users (api_key, username)
                VALUES (%s, %s)
                RETURNING id, api_key
                """,
                (api_key, username or "Anonymous")
            )
            user = result.fetchone()

        dm.close()
        return {"user_id": user[0], "api_key": user[1]}

    @staticmethod
    def validate_api_key(api_key):
        """Validate API key and return user_id"""
        dm = DataManager()

        with dm.engine.begin() as conn:
            result = conn.execute(
                "SELECT id, username FROM users WHERE api_key = %s",
                (api_key,)
            )
            user = result.fetchone()

        dm.close()

        if user:
            return {"user_id": user[0], "username": user[1]}
        return None

    @staticmethod
    def get_session_user():
        """Get current user from session"""
        if 'user_id' not in st.session_state:
            # For demo purposes, create anonymous user if none exists
            if 'api_key' not in st.session_state:
                # Check if there's a demo user
                dm = DataManager()
                with dm.engine.begin() as conn:
                    result = conn.execute(
                        "SELECT id, api_key FROM users WHERE username = 'demo' LIMIT 1"
                    )
                    user = result.fetchone()

                dm.close()

                if user:
                    st.session_state.user_id = user[0]
                    st.session_state.api_key = user[1]
                else:
                    # Create demo user
                    user_data = AuthManager.create_user("demo")
                    st.session_state.user_id = user_data['user_id']
                    st.session_state.api_key = user_data['api_key']

        return st.session_state.get('user_id')

    @staticmethod
    def require_auth():
        """Show auth UI and return user_id or None"""
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🔑 Authentication")

            if 'user_id' in st.session_state:
                user_id = st.session_state.user_id
                api_key = st.session_state.get('api_key', 'N/A')

                st.success(f"✅ Logged in (User #{user_id})")
                st.code(f"API Key: {api_key[:16]}...", language="text")

                if st.button("🔓 Logout"):
                    del st.session_state.user_id
                    del st.session_state.api_key
                    st.rerun()

                return user_id
            else:
                st.info("Enter API key or create new account")

                tab1, tab2 = st.tabs(["Login", "Sign Up"])

                with tab1:
                    api_key = st.text_input("API Key", type="password", key="login_key")
                    if st.button("🔐 Login", key="login_btn"):
                        user = AuthManager.validate_api_key(api_key)
                        if user:
                            st.session_state.user_id = user['user_id']
                            st.session_state.api_key = api_key
                            st.success(f"✅ Welcome {user['username']}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid API key")

                with tab2:
                    username = st.text_input("Username (optional)", key="signup_name")
                    if st.button("🆕 Create Account", key="signup_btn"):
                        user_data = AuthManager.create_user(username or None)
                        st.session_state.user_id = user_data['user_id']
                        st.session_state.api_key = user_data['api_key']

                        st.success("✅ Account created!")
                        st.code(f"API Key: {user_data['api_key']}", language="text")
                        st.warning("⚠️ Save this key! You won't see it again.")
                        st.rerun()

                return None


def save_custom_asset_to_db(ticker, asset_type, df, user_id):
    """Save custom asset to PostgreSQL with user isolation"""
    dm = DataManager()

    try:
        # Register asset as custom
        with dm.engine.begin() as conn:
            conn.execute(
                """
                INSERT INTO asset_metadata
                (ticker, asset_type, frequency, user_id, is_custom, last_update)
                VALUES (%s, %s, 'daily', %s, TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    is_custom = TRUE,
                    last_update = CURRENT_TIMESTAMP
                """,
                (ticker, asset_type.lower(), user_id)
            )

        # Prepare data
        df = df.copy()

        # Ensure timestamp column
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        elif 'Timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
        elif 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Time'])

        # Normalize column names
        column_mapping = {
            'Close': 'close',
            'Price': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Ensure required columns exist
        if 'close' not in df.columns:
            raise ValueError("CSV must have 'Close' or 'Price' column")

        # Fill missing OHLC with close price
        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                df[col] = df['close']

        if 'volume' not in df.columns:
            df['volume'] = 0

        # Select and insert data
        data_to_insert = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        data_to_insert['ticker'] = ticker
        data_to_insert['user_id'] = user_id

        # Bulk insert
        records = data_to_insert.to_dict('records')

        with dm.engine.begin() as conn:
            # Delete existing data for this user+ticker
            conn.execute(
                "DELETE FROM price_data WHERE ticker = %s AND user_id = %s",
                (ticker, user_id)
            )

            # Insert new data
            for record in records:
                conn.execute(
                    """
                    INSERT INTO price_data
                    (ticker, timestamp, open, high, low, close, volume, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, timestamp) DO NOTHING
                    """,
                    (
                        record['ticker'],
                        record['timestamp'],
                        record['open'],
                        record['high'],
                        record['low'],
                        record['close'],
                        record['volume'],
                        record['user_id']
                    )
                )

        dm.close()
        return True, len(records)

    except Exception as e:
        dm.close()
        return False, str(e)
