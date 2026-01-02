"""
Risk Configuration Manager
Allows users to customize risk tolerance and trading parameters
"""

import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class RiskProfile:
    """Risk profile configuration"""
    name: str

    # Portfolio allocation
    stock_allocation: float  # 0.0 to 1.0
    crypto_allocation: float  # 0.0 to 1.0

    # Position sizing
    max_position_size: float  # Max % of portfolio per position
    max_positions: int  # Max concurrent positions

    # Leverage
    max_leverage_stocks: float  # 1.0 to 3.0
    max_leverage_crypto: float  # 1.0 to 5.0

    # Stop losses
    stop_loss_pct_stocks: float  # 0.02 to 0.10
    stop_loss_pct_crypto: float  # 0.05 to 0.20

    # Risk limits
    max_daily_loss: float  # Max % loss per day
    max_drawdown: float  # Max portfolio drawdown

    # Model confidence thresholds
    min_confidence_stocks: float  # 0.50 to 0.95
    min_confidence_crypto: float  # 0.50 to 0.95

    # Trading frequency
    holding_period_days_stocks: int  # Min days to hold
    holding_period_days_crypto: int  # Min days to hold

    def validate(self) -> bool:
        """Validate configuration"""
        # Check allocations sum to 1.0
        total_allocation = self.stock_allocation + self.crypto_allocation
        if not (0.99 <= total_allocation <= 1.01):
            raise ValueError(f"Allocations must sum to 1.0, got {total_allocation}")

        # Check ranges
        if not (0.0 <= self.max_position_size <= 1.0):
            raise ValueError("max_position_size must be between 0 and 1")

        if self.max_leverage_stocks > 3.0:
            raise ValueError("max_leverage_stocks should not exceed 3.0")

        if self.max_leverage_crypto > 5.0:
            raise ValueError("max_leverage_crypto should not exceed 5.0")

        return True


class RiskManager:
    """
    Manages risk profiles and provides risk configuration

    Features:
    - Load/save risk profiles
    - Preset profiles (conservative, moderate, aggressive)
    - Custom profile creation
    - Real-time risk checks
    """

    def __init__(self, config_file: str = 'config/risk_profiles.json'):
        """Initialize risk manager"""
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or create default profiles
        self.profiles = self._load_profiles()

        # Current active profile
        self.active_profile: Optional[RiskProfile] = None

    def _load_profiles(self) -> Dict[str, RiskProfile]:
        """Load profiles from config file or create defaults"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return {
                    name: RiskProfile(**profile)
                    for name, profile in data.items()
                }
        else:
            # Create default profiles
            return self._create_default_profiles()

    def _create_default_profiles(self) -> Dict[str, RiskProfile]:
        """Create default risk profiles"""
        profiles = {
            'conservative': RiskProfile(
                name='conservative',
                stock_allocation=0.85,
                crypto_allocation=0.15,
                max_position_size=0.15,
                max_positions=3,
                max_leverage_stocks=1.0,
                max_leverage_crypto=1.0,
                stop_loss_pct_stocks=0.03,
                stop_loss_pct_crypto=0.06,
                max_daily_loss=0.015,
                max_drawdown=0.20,
                min_confidence_stocks=0.65,
                min_confidence_crypto=0.70,
                holding_period_days_stocks=10,
                holding_period_days_crypto=7
            ),
            'moderate': RiskProfile(
                name='moderate',
                stock_allocation=0.70,
                crypto_allocation=0.30,
                max_position_size=0.20,
                max_positions=5,
                max_leverage_stocks=1.5,
                max_leverage_crypto=2.0,
                stop_loss_pct_stocks=0.04,
                stop_loss_pct_crypto=0.08,
                max_daily_loss=0.025,
                max_drawdown=0.30,
                min_confidence_stocks=0.60,
                min_confidence_crypto=0.65,
                holding_period_days_stocks=7,
                holding_period_days_crypto=5
            ),
            'aggressive': RiskProfile(
                name='aggressive',
                stock_allocation=0.50,
                crypto_allocation=0.50,
                max_position_size=0.25,
                max_positions=8,
                max_leverage_stocks=2.0,
                max_leverage_crypto=3.0,
                stop_loss_pct_stocks=0.05,
                stop_loss_pct_crypto=0.10,
                max_daily_loss=0.04,
                max_drawdown=0.45,
                min_confidence_stocks=0.55,
                min_confidence_crypto=0.60,
                holding_period_days_stocks=5,
                holding_period_days_crypto=3
            )
        }

        # Save defaults
        self.save_profiles(profiles)
        return profiles

    def save_profiles(self, profiles: Optional[Dict[str, RiskProfile]] = None):
        """Save profiles to config file"""
        if profiles is None:
            profiles = self.profiles

        data = {
            name: asdict(profile)
            for name, profile in profiles.items()
        }

        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved {len(profiles)} risk profiles to {self.config_file}")

    def get_profile(self, name: str) -> RiskProfile:
        """Get a risk profile by name"""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found. Available: {list(self.profiles.keys())}")

        return self.profiles[name]

    def set_active_profile(self, name: str):
        """Set active risk profile"""
        profile = self.get_profile(name)
        profile.validate()
        self.active_profile = profile
        print(f"✓ Active risk profile: {name}")
        return profile

    def create_custom_profile(self, name: str, **kwargs) -> RiskProfile:
        """
        Create a custom risk profile

        Example:
            rm.create_custom_profile(
                'my_profile',
                stock_allocation=0.60,
                crypto_allocation=0.40,
                max_leverage_crypto=2.5
            )
        """
        # Start with moderate as base
        base = self.profiles['moderate']
        base_dict = asdict(base)

        # Update with custom values
        base_dict['name'] = name
        base_dict.update(kwargs)

        # Create and validate
        profile = RiskProfile(**base_dict)
        profile.validate()

        # Save
        self.profiles[name] = profile
        self.save_profiles()

        print(f"✓ Created custom profile: {name}")
        return profile

    def list_profiles(self):
        """List all available profiles"""
        print("\nAvailable Risk Profiles:")
        print("=" * 70)

        for name, profile in self.profiles.items():
            print(f"\n{name.upper()}")
            print(f"  Allocation: {profile.stock_allocation*100:.0f}% stocks, {profile.crypto_allocation*100:.0f}% crypto")
            print(f"  Max leverage: {profile.max_leverage_stocks}x stocks, {profile.max_leverage_crypto}x crypto")
            print(f"  Stop losses: {profile.stop_loss_pct_stocks*100:.1f}% stocks, {profile.stop_loss_pct_crypto*100:.1f}% crypto")
            print(f"  Max daily loss: {profile.max_daily_loss*100:.1f}%")
            print(f"  Max drawdown: {profile.max_drawdown*100:.1f}%")

    def check_position_size(self, position_value: float, portfolio_value: float) -> bool:
        """Check if position size is within limits"""
        if self.active_profile is None:
            raise ValueError("No active risk profile set")

        position_pct = position_value / portfolio_value
        return position_pct <= self.active_profile.max_position_size

    def check_daily_loss(self, loss_pct: float) -> bool:
        """Check if daily loss exceeds limit"""
        if self.active_profile is None:
            raise ValueError("No active risk profile set")

        return abs(loss_pct) <= self.active_profile.max_daily_loss

    def check_drawdown(self, current_value: float, peak_value: float) -> bool:
        """Check if drawdown exceeds limit"""
        if self.active_profile is None:
            raise ValueError("No active risk profile set")

        drawdown = (peak_value - current_value) / peak_value
        return drawdown <= self.active_profile.max_drawdown

    def get_position_size(self, portfolio_value: float, confidence: float = 1.0) -> float:
        """
        Calculate position size based on risk profile and model confidence

        Args:
            portfolio_value: Current portfolio value
            confidence: Model confidence (0.0 to 1.0)

        Returns:
            Position size in dollars
        """
        if self.active_profile is None:
            raise ValueError("No active risk profile set")

        # Base position size
        base_size = portfolio_value * self.active_profile.max_position_size

        # Scale by confidence
        scaled_size = base_size * confidence

        return scaled_size


def demo_risk_manager():
    """Demonstrate risk manager usage"""
    print("\n" + "="*70)
    print("RISK MANAGER DEMO")
    print("="*70)

    # Initialize
    rm = RiskManager('config/risk_profiles.json')

    # List profiles
    rm.list_profiles()

    # Set active profile
    print("\n" + "-"*70)
    print("Setting 'moderate' as active profile:")
    rm.set_active_profile('moderate')

    profile = rm.active_profile
    print(f"\nActive Profile Settings:")
    print(f"  Stock allocation: {profile.stock_allocation*100:.0f}%")
    print(f"  Crypto allocation: {profile.crypto_allocation*100:.0f}%")
    print(f"  Max position size: {profile.max_position_size*100:.0f}%")
    print(f"  Max positions: {profile.max_positions}")

    # Create custom profile
    print("\n" + "-"*70)
    print("Creating custom profile:")
    custom = rm.create_custom_profile(
        'my_custom',
        stock_allocation=0.65,
        crypto_allocation=0.35,
        max_leverage_crypto=2.5,
        stop_loss_pct_crypto=0.09
    )

    # Risk checks
    print("\n" + "-"*70)
    print("Risk Checks (with moderate profile):")

    portfolio_value = 100000
    position_value = 15000

    ok = rm.check_position_size(position_value, portfolio_value)
    print(f"  Position size check (${position_value:,} on ${portfolio_value:,}): {'✓ OK' if ok else '✗ TOO LARGE'}")

    daily_loss = -0.02
    ok = rm.check_daily_loss(daily_loss)
    print(f"  Daily loss check ({daily_loss*100:.1f}%): {'✓ OK' if ok else '✗ EXCEEDED'}")

    current_value = 85000
    peak_value = 100000
    ok = rm.check_drawdown(current_value, peak_value)
    drawdown = (peak_value - current_value) / peak_value
    print(f"  Drawdown check ({drawdown*100:.1f}%): {'✓ OK' if ok else '✗ EXCEEDED'}")

    # Position sizing
    print("\n" + "-"*70)
    print("Position Sizing Examples:")

    for confidence in [0.60, 0.75, 0.90]:
        size = rm.get_position_size(portfolio_value, confidence)
        print(f"  Confidence {confidence*100:.0f}%: ${size:,.0f} position")


if __name__ == '__main__':
    demo_risk_manager()
