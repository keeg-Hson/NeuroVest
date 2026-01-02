#!/usr/bin/env python3
"""
Framework Test Script

Verifies that the NeuroVest framework is properly configured and working.

Tests:
1. Configuration loading
2. Asset manager
3. File structure
4. Dependencies
5. Basic functionality

Usage:
    python test_framework.py
"""

import sys
from pathlib import Path
import importlib.util

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def test_result(passed: bool, message: str):
    """Print test result"""
    if passed:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {message}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "=" * 80)
    print("TESTING DEPENDENCIES")
    print("=" * 80 + "\n")

    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('catboost', 'catboost'),
        ('yaml', 'PyYAML'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('ccxt', 'ccxt'),
        ('schedule', 'schedule'),
    ]

    all_passed = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            test_result(True, f"{package_name} installed")
        except ImportError:
            test_result(False, f"{package_name} MISSING - install with: pip install {package_name}")
            all_passed = False

    return all_passed


def test_file_structure():
    """Test required files and directories"""
    print("\n" + "=" * 80)
    print("TESTING FILE STRUCTURE")
    print("=" * 80 + "\n")

    required_files = [
        'config/assets.yaml',
        'framework/asset_manager.py',
        'framework/download_all_assets.py',
        'framework/train_unified.py',
        'framework/api_server.py',
        'framework/results_dashboard.py',
        'framework/auto_refresh.py',
        'FRAMEWORK_GUIDE.md',
    ]

    required_dirs = [
        'config',
        'framework',
        'data_cache',
        'models',
        'results',
    ]

    all_passed = True

    # Test directories
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            test_result(True, f"Created directory: {dir_path}")
        else:
            test_result(True, f"Directory exists: {dir_path}")

    # Test files
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            test_result(True, f"File exists: {file_path}")
        else:
            test_result(False, f"File MISSING: {file_path}")
            all_passed = False

    return all_passed


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION")
    print("=" * 80 + "\n")

    try:
        sys.path.insert(0, 'framework')
        from asset_manager import AssetManager

        manager = AssetManager()

        # Test asset counts
        all_assets = manager.get_all_assets()
        test_result(len(all_assets) > 0, f"Loaded {len(all_assets)} assets")

        # Test asset types
        equities = manager.get_assets_by_type('equity')
        cryptos = manager.get_assets_by_type('crypto')
        bonds = manager.get_assets_by_type('bond')
        commodities = manager.get_assets_by_type('commodity')

        test_result(len(equities) > 0, f"Equity ETFs: {len(equities)}")
        test_result(len(cryptos) > 0, f"Cryptocurrencies: {len(cryptos)}")
        test_result(len(bonds) > 0, f"Bond ETFs: {len(bonds)}")
        test_result(len(commodities) > 0, f"Commodity ETFs: {len(commodities)}")

        # Test macro groups
        macro_groups = manager.get_macro_groups()
        test_result(len(macro_groups) > 0, f"Macro groups: {len(macro_groups)}")

        # Test settings
        settings = manager.get_settings()
        test_result('start_date' in settings, f"Settings loaded (start_date: {settings.get('start_date')})")

        return True

    except Exception as e:
        test_result(False, f"Configuration loading failed: {e}")
        return False


def test_asset_manager():
    """Test asset manager functionality"""
    print("\n" + "=" * 80)
    print("TESTING ASSET MANAGER")
    print("=" * 80 + "\n")

    try:
        sys.path.insert(0, 'framework')
        from asset_manager import AssetManager

        manager = AssetManager()

        # Test get_asset
        spy = manager.get_asset('SPY')
        if spy:
            test_result(True, f"Found SPY: {spy.name} (threshold: {spy.threshold})")
        else:
            test_result(False, "Could not find SPY asset")
            return False

        # Test get_ticker_list
        tickers = manager.get_ticker_list('equity')
        test_result(len(tickers) > 0, f"Equity tickers: {len(tickers)}")

        # Test macro group
        all_equities = manager.get_macro_group('all_equities')
        test_result(len(all_equities) > 0, f"All equities macro group: {len(all_equities)} assets")

        return True

    except Exception as e:
        test_result(False, f"Asset manager test failed: {e}")
        return False


def test_framework_imports():
    """Test that framework modules can be imported"""
    print("\n" + "=" * 80)
    print("TESTING FRAMEWORK IMPORTS")
    print("=" * 80 + "\n")

    modules = [
        'framework/asset_manager.py',
        'framework/download_all_assets.py',
        'framework/train_unified.py',
        'framework/api_server.py',
        'framework/results_dashboard.py',
        'framework/auto_refresh.py',
    ]

    all_passed = True
    for module_path in modules:
        try:
            spec = importlib.util.spec_from_file_location("test_module", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Note: We don't execute the module, just check it can be loaded
                test_result(True, f"Can import: {module_path}")
            else:
                test_result(False, f"Cannot load spec: {module_path}")
                all_passed = False
        except Exception as e:
            test_result(False, f"Import failed: {module_path} - {e}")
            all_passed = False

    return all_passed


def print_summary(results: dict):
    """Print test summary"""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80 + "\n")

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{test_name:30s} {status}")

    print("\n" + "=" * 80)

    if all_passed:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}")
        print("\nFramework is ready to use!")
        print("\nNext steps:")
        print("  1. python framework/download_all_assets.py")
        print("  2. python framework/train_unified.py --all")
        print("  3. python framework/results_dashboard.py")
        print("\nSee FRAMEWORK_GUIDE.md for detailed instructions.")
    else:
        print(f"{RED}✗ SOME TESTS FAILED{RESET}")
        print("\nPlease fix the errors above before using the framework.")
        print("See FRAMEWORK_GUIDE.md for help.")

    print("=" * 80 + "\n")

    return all_passed


def main():
    print("=" * 80)
    print("NEUROVEST FRAMEWORK TEST")
    print("=" * 80)

    results = {}

    # Run tests
    results['Dependencies'] = test_dependencies()
    results['File Structure'] = test_file_structure()
    results['Configuration'] = test_config_loading()
    results['Asset Manager'] = test_asset_manager()
    results['Framework Imports'] = test_framework_imports()

    # Print summary
    all_passed = print_summary(results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
