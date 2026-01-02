"""
NeuroVest Trading API
REST API for the trading system using FastAPI
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk_manager import RiskManager, RiskProfile
from core.data_manager_postgres import DataManager


# ==============================================================================
# API Models
# ==============================================================================

class AssetType(str, Enum):
    """Asset type enumeration"""
    STOCK = "stock"
    CRYPTO = "crypto"


class TradingMode(str, Enum):
    """Trading mode"""
    PAPER = "paper"
    LIVE = "live"


class SignalType(str, Enum):
    """Signal type"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class RiskLevel(str, Enum):
    """Risk level presets"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class PortfolioStatus(BaseModel):
    """Portfolio status response"""
    total_value: float
    cash: float
    stock_value: float
    crypto_value: float
    positions: int
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    updated_at: datetime


class Position(BaseModel):
    """Position information"""
    ticker: str
    asset_type: AssetType
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    days_held: int


class Signal(BaseModel):
    """Trading signal"""
    ticker: str
    asset_type: AssetType
    signal: SignalType
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float
    reason: str
    timestamp: datetime


class Trade(BaseModel):
    """Trade execution"""
    ticker: str
    asset_type: AssetType
    action: str  # buy, sell
    quantity: float
    price: float
    total_value: float
    timestamp: datetime
    mode: TradingMode
    status: str  # executed, pending, failed


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_profit_per_trade: float
    best_trade: float
    worst_trade: float


class RiskProfileUpdate(BaseModel):
    """Risk profile update request"""
    name: Optional[str] = None
    stock_allocation: Optional[float] = Field(None, ge=0.0, le=1.0)
    crypto_allocation: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_leverage_stocks: Optional[float] = Field(None, ge=1.0, le=3.0)
    max_leverage_crypto: Optional[float] = Field(None, ge=1.0, le=5.0)
    stop_loss_pct_stocks: Optional[float] = Field(None, ge=0.01, le=0.10)
    stop_loss_pct_crypto: Optional[float] = Field(None, ge=0.01, le=0.20)
    max_daily_loss: Optional[float] = Field(None, ge=0.005, le=0.10)


class DataQuery(BaseModel):
    """Data query parameters"""
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="NeuroVest Trading API",
    description="AI-powered algorithmic trading system API",
    version="1.0.0"
)

security = HTTPBearer()


# ==============================================================================
# Dependencies
# ==============================================================================

def get_risk_manager():
    """Get risk manager instance"""
    return RiskManager()


def get_data_manager():
    """Get data manager instance"""
    return DataManager()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if token != "demo_token_replace_in_production":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token


# ==============================================================================
# Health & Status Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "NeuroVest Trading API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ==============================================================================
# Portfolio Endpoints
# ==============================================================================

@app.get("/portfolio/status", response_model=PortfolioStatus)
async def get_portfolio_status(token: str = Depends(verify_token)):
    """Get current portfolio status"""
    # In production, fetch from database
    return PortfolioStatus(
        total_value=154592.50,
        cash=50000.00,
        stock_value=85000.00,
        crypto_value=19592.50,
        positions=5,
        daily_pnl=2341.20,
        daily_pnl_pct=1.54,
        total_pnl=54592.50,
        total_pnl_pct=54.59,
        updated_at=datetime.now()
    )


@app.get("/portfolio/positions", response_model=List[Position])
async def get_positions(token: str = Depends(verify_token)):
    """Get all open positions"""
    # In production, fetch from database
    return [
        Position(
            ticker="SPY",
            asset_type=AssetType.STOCK,
            quantity=100,
            entry_price=450.00,
            current_price=465.50,
            market_value=46550.00,
            unrealized_pnl=1550.00,
            unrealized_pnl_pct=3.44,
            entry_time=datetime.now(),
            days_held=5
        )
    ]


@app.get("/portfolio/performance", response_model=PerformanceMetrics)
async def get_performance(token: str = Depends(verify_token)):
    """Get portfolio performance metrics"""
    return PerformanceMetrics(
        total_return=0.5459,
        annualized_return=0.2122,
        sharpe_ratio=7.47,
        max_drawdown=-0.2740,
        win_rate=0.535,
        total_trades=475,
        avg_profit_per_trade=0.0347,
        best_trade=0.1250,
        worst_trade=-0.0850
    )


# ==============================================================================
# Trading Endpoints
# ==============================================================================

@app.get("/signals", response_model=List[Signal])
async def get_signals(
    asset_type: Optional[AssetType] = None,
    min_confidence: float = 0.60,
    token: str = Depends(verify_token)
):
    """Get current trading signals"""
    # In production, generate from ML models
    signals = [
        Signal(
            ticker="SPY",
            asset_type=AssetType.STOCK,
            signal=SignalType.BUY,
            confidence=0.87,
            entry_price=465.50,
            target_price=485.00,
            stop_loss=450.00,
            position_size=15000.00,
            reason="Strong bullish momentum, high model confidence",
            timestamp=datetime.now()
        ),
        Signal(
            ticker="BTC_USDT",
            asset_type=AssetType.CRYPTO,
            signal=SignalType.BUY,
            confidence=0.75,
            entry_price=65000.00,
            target_price=72000.00,
            stop_loss=60000.00,
            position_size=8000.00,
            reason="Breakout pattern detected",
            timestamp=datetime.now()
        )
    ]

    # Filter by asset type
    if asset_type:
        signals = [s for s in signals if s.asset_type == asset_type]

    # Filter by confidence
    signals = [s for s in signals if s.confidence >= min_confidence]

    return signals


@app.post("/trade", response_model=Trade)
async def execute_trade(
    ticker: str,
    action: str,
    quantity: float,
    mode: TradingMode = TradingMode.PAPER,
    token: str = Depends(verify_token)
):
    """Execute a trade"""
    if action not in ['buy', 'sell']:
        raise HTTPException(status_code=400, detail="Action must be 'buy' or 'sell'")

    # In production, execute via broker API
    return Trade(
        ticker=ticker,
        asset_type=AssetType.STOCK,  # Determine from ticker
        action=action,
        quantity=quantity,
        price=465.50,  # Get current price
        total_value=quantity * 465.50,
        timestamp=datetime.now(),
        mode=mode,
        status="executed" if mode == TradingMode.PAPER else "pending"
    )


# ==============================================================================
# Risk Management Endpoints
# ==============================================================================

@app.get("/risk/profiles")
async def list_risk_profiles(token: str = Depends(verify_token)):
    """List all available risk profiles"""
    rm = get_risk_manager()
    return {
        "profiles": list(rm.profiles.keys()),
        "active": rm.active_profile.name if rm.active_profile else None
    }


@app.get("/risk/profile/{profile_name}")
async def get_risk_profile(
    profile_name: str,
    token: str = Depends(verify_token)
):
    """Get a specific risk profile"""
    rm = get_risk_manager()
    try:
        profile = rm.get_profile(profile_name)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/risk/profile/active")
async def set_active_risk_profile(
    profile_name: str,
    token: str = Depends(verify_token)
):
    """Set the active risk profile"""
    rm = get_risk_manager()
    try:
        profile = rm.set_active_profile(profile_name)
        return {
            "message": f"Active profile set to {profile_name}",
            "profile": profile
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/risk/profile/create")
async def create_custom_risk_profile(
    update: RiskProfileUpdate,
    token: str = Depends(verify_token)
):
    """Create a custom risk profile"""
    if not update.name:
        raise HTTPException(status_code=400, detail="Profile name is required")

    rm = get_risk_manager()

    # Build kwargs from non-None values
    kwargs = {k: v for k, v in update.dict().items() if v is not None and k != 'name'}

    try:
        profile = rm.create_custom_profile(update.name, **kwargs)
        return {
            "message": f"Created custom profile: {update.name}",
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# Data Endpoints
# ==============================================================================

@app.get("/data/{ticker}")
async def get_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Get market data for a specific ticker"""
    dm = get_data_manager()

    df = dm.get_data(ticker, start_date, end_date)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    # Convert to JSON-serializable format
    data = df.reset_index().to_dict(orient='records')

    return {
        "ticker": ticker,
        "records": len(data),
        "start_date": df.index[0].isoformat(),
        "end_date": df.index[-1].isoformat(),
        "data": data[:100]  # Limit to first 100 for API response
    }


@app.get("/data/stats")
async def get_data_stats(token: str = Depends(verify_token)):
    """Get database statistics"""
    dm = get_data_manager()
    stats = dm.get_stats()
    return stats


@app.post("/data/update")
async def trigger_data_update(
    background_tasks: BackgroundTasks,
    ticker: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Trigger data update for specific ticker or all assets"""
    # In production, trigger update job
    background_tasks.add_task(update_data_task, ticker)

    return {
        "message": "Data update triggered",
        "ticker": ticker or "all",
        "status": "processing"
    }


# ==============================================================================
# Background Tasks
# ==============================================================================

async def update_data_task(ticker: Optional[str] = None):
    """Background task to update data"""
    # Implement actual data update logic
    pass


# ==============================================================================
# WebSocket for Real-time Updates
# ==============================================================================

from fastapi import WebSocket
from typing import List as TypingList

active_connections: TypingList[WebSocket] = []


@app.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # In production, send real-time updates
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "portfolio_update",
                "data": {
                    "total_value": 154592.50,
                    "timestamp": datetime.now().isoformat()
                }
            })
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)


# ==============================================================================
# Run Server
# ==============================================================================

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
