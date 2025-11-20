#!/usr/bin/env python3
"""
Email Newsletter Generator

Generates and sends market analysis newsletters with noteworthy findings.

Usage:
    python3 newsletter_generator.py --preview    # Preview without sending
    python3 newsletter_generator.py --send       # Send email
    python3 newsletter_generator.py --assets SPY,BTC/USDT

Environment variables:
    SMTP_HOST - SMTP server host (default: smtp.gmail.com)
    SMTP_PORT - SMTP server port (default: 587)
    SMTP_USER - SMTP username/email
    SMTP_PASSWORD - SMTP password or app password
    NEWSLETTER_RECIPIENTS - Comma-separated list of recipient emails
"""

import os
import smtplib
import argparse
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timedelta

def load_predictions():
    """Load all predictions"""
    pred_path = Path("logs/daily_predictions.csv")
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_backtest_results():
    """Load latest backtest results"""
    latest_path = Path("logs/latest.json")
    if latest_path.exists():
        import json
        with open(latest_path) as f:
            return json.load(f)
    return None

def load_asset_data(asset):
    """Load price data for an asset"""
    if asset == "SPY":
        data_path = Path("data/SPY.csv")
    else:
        asset_file = asset.replace("/", "_") + "_1d.csv"
        data_path = Path(f"data_cache/{asset_file}")

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def find_noteworthy_signals(predictions, lookback_days=7):
    """Find noteworthy signals from recent predictions"""
    if predictions is None:
        return []

    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent = predictions[predictions['Date'] >= cutoff]

    noteworthy = []

    for _, row in recent.iterrows():
        pred = row.get('Prediction', 1)
        crash_conf = row.get('Crash_Conf', 0)
        spike_conf = row.get('Spike_Conf', 0)

        # High confidence signals
        if pred == 0 and crash_conf > 0.7:
            noteworthy.append({
                'date': row['Date'],
                'type': 'CRASH',
                'confidence': crash_conf,
                'description': f"High confidence CRASH signal ({crash_conf:.1%})"
            })
        elif pred == 2 and spike_conf > 0.7:
            noteworthy.append({
                'date': row['Date'],
                'type': 'SPIKE',
                'confidence': spike_conf,
                'description': f"High confidence SPIKE signal ({spike_conf:.1%})"
            })

    return noteworthy

def calculate_performance_summary(asset):
    """Calculate performance metrics for an asset"""
    df = load_asset_data(asset)
    if df is None:
        return None

    latest = df.iloc[-1]
    summary = {
        'asset': asset,
        'latest_price': latest['Close'],
        'latest_date': latest['Date'],
    }

    # Calculate returns
    if len(df) >= 5:
        summary['5d_return'] = (latest['Close'] / df.iloc[-5]['Close'] - 1) * 100
    if len(df) >= 20:
        summary['20d_return'] = (latest['Close'] / df.iloc[-20]['Close'] - 1) * 100
    if len(df) >= 252:
        summary['1y_return'] = (latest['Close'] / df.iloc[-252]['Close'] - 1) * 100

    return summary

def generate_newsletter_html(assets=None):
    """Generate newsletter content in HTML format"""
    if assets is None:
        assets = ['SPY']

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .signal {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .signal-crash {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
            .signal-spike {{ background-color: #e8f5e9; border-left: 4px solid #4caf50; }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>📊 NeuroVest Market Newsletter</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
    """

    # Load predictions
    predictions = load_predictions()

    # Asset summaries
    html += "<h2>📈 Asset Performance Summary</h2>"
    html += "<table><tr><th>Asset</th><th>Price</th><th>5D Return</th><th>20D Return</th><th>1Y Return</th></tr>"

    for asset in assets:
        summary = calculate_performance_summary(asset)
        if summary:
            def format_return(val):
                if val is None:
                    return "N/A"
                cls = "positive" if val >= 0 else "negative"
                return f'<span class="{cls}">{val:+.1f}%</span>'

            html += f"""
            <tr>
                <td><strong>{asset}</strong></td>
                <td>${summary.get('latest_price', 0):.2f}</td>
                <td>{format_return(summary.get('5d_return'))}</td>
                <td>{format_return(summary.get('20d_return'))}</td>
                <td>{format_return(summary.get('1y_return'))}</td>
            </tr>
            """

    html += "</table>"

    # Noteworthy signals
    signals = find_noteworthy_signals(predictions)
    if signals:
        html += "<h2>⚡ Noteworthy Signals (Last 7 Days)</h2>"
        for sig in signals:
            cls = "signal-crash" if sig['type'] == 'CRASH' else "signal-spike"
            html += f"""
            <div class="signal {cls}">
                <strong>{sig['date'].strftime('%Y-%m-%d')}</strong>: {sig['description']}
            </div>
            """
    else:
        html += "<h2>⚡ Noteworthy Signals</h2>"
        html += "<p>No high-confidence signals in the last 7 days.</p>"

    # Backtest performance
    backtest = load_backtest_results()
    if backtest:
        metrics = backtest.get('metrics', {})
        html += "<h2>🎯 Latest Backtest Performance</h2>"
        html += f"""
        <div class="metric">
            <div class="metric-value">{metrics.get('total_return', 0)*100:.1f}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('sharpe', 0):.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('win_rate', 0)*100:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics.get('trades', 0)}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        """

    # Footer
    html += """
        <div class="footer">
            <p>This newsletter is generated automatically by NeuroVest.
            Past performance does not guarantee future results.
            This is not financial advice.</p>
        </div>
    </body>
    </html>
    """

    return html

def generate_newsletter_text(assets=None):
    """Generate newsletter content in plain text format"""
    if assets is None:
        assets = ['SPY']

    text = f"""
NEUROVEST MARKET NEWSLETTER
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
{'='*60}

ASSET PERFORMANCE SUMMARY
{'-'*60}
"""

    for asset in assets:
        summary = calculate_performance_summary(asset)
        if summary:
            text += f"""
{asset}:
  Price: ${summary.get('latest_price', 0):.2f}
  5D Return: {summary.get('5d_return', 0):+.1f}%
  20D Return: {summary.get('20d_return', 0):+.1f}%
  1Y Return: {summary.get('1y_return', 0):+.1f}%
"""

    # Noteworthy signals
    predictions = load_predictions()
    signals = find_noteworthy_signals(predictions)

    text += f"""
{'='*60}
NOTEWORTHY SIGNALS (Last 7 Days)
{'-'*60}
"""
    if signals:
        for sig in signals:
            text += f"  {sig['date'].strftime('%Y-%m-%d')}: {sig['description']}\n"
    else:
        text += "  No high-confidence signals in the last 7 days.\n"

    # Backtest
    backtest = load_backtest_results()
    if backtest:
        metrics = backtest.get('metrics', {})
        text += f"""
{'='*60}
LATEST BACKTEST PERFORMANCE
{'-'*60}
  Total Return: {metrics.get('total_return', 0)*100:.1f}%
  Sharpe Ratio: {metrics.get('sharpe', 0):.2f}
  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
  Total Trades: {metrics.get('trades', 0)}
"""

    text += f"""
{'='*60}
This newsletter is generated automatically by NeuroVest.
Past performance does not guarantee future results.
This is not financial advice.
"""

    return text

def send_email(subject, html_content, text_content):
    """Send newsletter via email"""
    smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    recipients = os.getenv('NEWSLETTER_RECIPIENTS', '').split(',')

    if not smtp_user or not smtp_password:
        print("❌ SMTP credentials not set. Export SMTP_USER and SMTP_PASSWORD.")
        return False

    if not recipients or recipients == ['']:
        print("❌ No recipients set. Export NEWSLETTER_RECIPIENTS.")
        return False

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = ', '.join(recipients)

    msg.attach(MIMEText(text_content, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipients, msg.as_string())
        server.quit()
        print(f"✅ Newsletter sent to {len(recipients)} recipient(s)")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate and send market newsletter")
    parser.add_argument("--preview", action="store_true", help="Preview newsletter without sending")
    parser.add_argument("--send", action="store_true", help="Send newsletter via email")
    parser.add_argument("--assets", default="SPY", help="Comma-separated list of assets")
    parser.add_argument("--output", help="Save HTML to file")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(',')]

    print("=" * 60)
    print("  NEUROVEST NEWSLETTER GENERATOR")
    print("=" * 60)

    html_content = generate_newsletter_html(assets)
    text_content = generate_newsletter_text(assets)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(html_content)
        print(f"✅ Saved to {args.output}")

    if args.preview:
        print("\n" + text_content)

    if args.send:
        subject = f"NeuroVest Market Newsletter - {datetime.now().strftime('%Y-%m-%d')}"
        send_email(subject, html_content, text_content)

    if not args.preview and not args.send and not args.output:
        print("\nUsage:")
        print("  --preview  : View newsletter in terminal")
        print("  --send     : Send via email (requires SMTP env vars)")
        print("  --output   : Save HTML to file")

if __name__ == "__main__":
    main()
